import os
import time

import numpy as np
import torch
import torch.nn as nn

from prepare import prepare_ctr_data, load_ctr_data

TIME_BUDGET = int(os.environ.get("CTR_TIME_BUDGET", "300"))


def _parse_env_list(name):
    value = os.environ.get(name, "")
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_mlp(input_dim, hidden_dims, dropout):
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def _build_mlp_stack(input_dim, hidden_dims, dropout):
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h
    return nn.Sequential(*layers), in_dim


class LRModel(nn.Module):
    def __init__(self, cat_dims, num_dim):
        super().__init__()
        self.cat_dims = cat_dims
        self.num_dim = num_dim
        self.cat_emb = nn.ModuleList([nn.Embedding(dim, 1) for dim in cat_dims]) if cat_dims else nn.ModuleList()
        self.num_linear = nn.Linear(num_dim, 1, bias=False) if num_dim > 0 else None
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_cat, x_num):
        out = self.bias
        if len(self.cat_emb) > 0:
            emb = torch.stack([self.cat_emb[i](x_cat[:, i]) for i in range(len(self.cat_emb))], dim=0)
            out = out + emb.sum(dim=0)
        if self.num_linear is not None and x_num.shape[1] > 0:
            out = out + self.num_linear(x_num)
        return out


class FMModel(nn.Module):
    def __init__(self, cat_dims, num_dim, embed_dim):
        super().__init__()
        self.linear = LRModel(cat_dims, num_dim)
        self.cat_emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims]) if cat_dims else nn.ModuleList()

    def forward(self, x_cat, x_num):
        linear_out = self.linear(x_cat, x_num)
        if len(self.cat_emb) == 0:
            return linear_out
        embs = torch.stack([self.cat_emb[i](x_cat[:, i]) for i in range(len(self.cat_emb))], dim=1)
        sum_emb = embs.sum(dim=1)
        sum_square = sum_emb * sum_emb
        square_sum = (embs * embs).sum(dim=1)
        fm_out = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)
        return linear_out + fm_out


class DNNModel(nn.Module):
    def __init__(self, cat_dims, num_dim, embed_dim, hidden_dims, dropout):
        super().__init__()
        self.cat_dims = cat_dims
        self.num_dim = num_dim
        self.embed_dim = embed_dim
        self.cat_emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims]) if cat_dims else nn.ModuleList()
        input_dim = len(cat_dims) * embed_dim + num_dim
        self.mlp = _build_mlp(input_dim, hidden_dims, dropout) if input_dim > 0 else None

    def forward(self, x_cat, x_num):
        if self.mlp is None:
            return torch.zeros((x_cat.shape[0], 1), device=x_cat.device)
        if len(self.cat_emb) > 0:
            embs = torch.stack([self.cat_emb[i](x_cat[:, i]) for i in range(len(self.cat_emb))], dim=1)
            flat = embs.flatten(1)
        else:
            flat = torch.zeros((x_cat.shape[0], 0), device=x_cat.device)
        if x_num.shape[1] > 0:
            dnn_in = torch.cat([flat, x_num], dim=1)
        else:
            dnn_in = flat
        return self.mlp(dnn_in)


class DeepFMModel(nn.Module):
    def __init__(self, cat_dims, num_dim, embed_dim, hidden_dims, dropout):
        super().__init__()
        self.linear = LRModel(cat_dims, num_dim)
        self.cat_emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims]) if cat_dims else nn.ModuleList()
        input_dim = len(cat_dims) * embed_dim + num_dim
        self.mlp = _build_mlp(input_dim, hidden_dims, dropout) if input_dim > 0 else None

    def forward(self, x_cat, x_num):
        linear_out = self.linear(x_cat, x_num)
        if len(self.cat_emb) > 0:
            embs = torch.stack([self.cat_emb[i](x_cat[:, i]) for i in range(len(self.cat_emb))], dim=1)
            sum_emb = embs.sum(dim=1)
            sum_square = sum_emb * sum_emb
            square_sum = (embs * embs).sum(dim=1)
            fm_out = 0.5 * (sum_square - square_sum).sum(dim=1, keepdim=True)
            flat = embs.flatten(1)
        else:
            fm_out = torch.zeros((x_cat.shape[0], 1), device=x_cat.device)
            flat = torch.zeros((x_cat.shape[0], 0), device=x_cat.device)
        if x_num.shape[1] > 0:
            dnn_in = torch.cat([flat, x_num], dim=1)
        else:
            dnn_in = flat
        dnn_out = self.mlp(dnn_in) if self.mlp is not None else torch.zeros((x_cat.shape[0], 1), device=x_cat.device)
        return linear_out + fm_out + dnn_out


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.kernels = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for i in range(self.num_layers):
            # x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l
            xw = self.kernels[i](x)  # [B, 1]
            x = x0 * xw + self.bias[i] + x
        return x


class DCNModel(nn.Module):
    def __init__(self, cat_dims, num_dim, embed_dim, hidden_dims, dropout, cross_layers):
        super().__init__()
        self.cat_emb = nn.ModuleList([nn.Embedding(dim, embed_dim) for dim in cat_dims]) if cat_dims else nn.ModuleList()
        input_dim = len(cat_dims) * embed_dim + num_dim
        self.cross = CrossNet(input_dim, cross_layers) if cross_layers > 0 and input_dim > 0 else None
        self.deep, deep_out_dim = _build_mlp_stack(input_dim, hidden_dims, dropout) if input_dim > 0 else (None, 0)
        self.final = nn.Linear((input_dim if self.cross is not None else 0) + deep_out_dim, 1)

    def forward(self, x_cat, x_num):
        if len(self.cat_emb) > 0:
            embs = torch.stack([self.cat_emb[i](x_cat[:, i]) for i in range(len(self.cat_emb))], dim=1)
            flat = embs.flatten(1)
        else:
            flat = torch.zeros((x_cat.shape[0], 0), device=x_cat.device)
        dnn_in = torch.cat([flat, x_num], dim=1) if x_num.shape[1] > 0 else flat
        cross_out = self.cross(dnn_in) if self.cross is not None else torch.zeros((dnn_in.shape[0], 0), device=dnn_in.device)
        deep_out = self.deep(dnn_in) if self.deep is not None else torch.zeros((dnn_in.shape[0], 0), device=dnn_in.device)
        x = torch.cat([cross_out, deep_out], dim=1) if cross_out.shape[1] + deep_out.shape[1] > 0 else torch.zeros((dnn_in.shape[0], 0), device=dnn_in.device)
        return self.final(x)


def _auc_score(y_true, y_score):
    y_true = y_true.astype(np.int64)
    order = np.argsort(y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = np.arange(1, len(y_true) + 1)
    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _logloss(y_true, y_prob):
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def run_ctr():
    t_start = time.time()
    data_path = os.environ.get("CTR_DATA_PATH")
    if not data_path:
        raise SystemExit("CTR_DATA_PATH is required")
    feature_config_path = os.environ.get("CTR_FEATURE_CONFIG", "")
    test_ratio = float(os.environ.get("CTR_TEST_RATIO", "0.1"))
    seed = int(os.environ.get("CTR_SEED", "42"))
    force = os.environ.get("CTR_FORCE_PREPARE", "0") == "1"
    prepare_ctr_data(
        data_path=data_path,
        test_ratio=test_ratio,
        seed=seed,
        force=force,
        feature_config_path=feature_config_path or None,
    )
    meta, train_data, val_data = load_ctr_data(data_path, feature_config_path or None)
    cat_dims = meta["cat_dims"]
    num_dim = len(meta["num_cols"])
    model_type = os.environ.get("CTR_MODEL", "DeepFM").upper()
    embed_dim = int(os.environ.get("CTR_EMBED_DIM", "16"))
    hidden_dims = _parse_env_list("CTR_HIDDEN_DIMS")
    hidden_dims = [int(x) for x in hidden_dims] if hidden_dims else [256, 128]
    dropout = float(os.environ.get("CTR_DROPOUT", "0.0"))
    cross_layers = int(os.environ.get("CTR_CROSS_LAYERS", "3"))
    lr = float(os.environ.get("CTR_LR", "0.001"))
    weight_decay = float(os.environ.get("CTR_WEIGHT_DECAY", "0.0"))
    batch_size = int(os.environ.get("CTR_BATCH_SIZE", "4096"))
    eval_batch_size = int(os.environ.get("CTR_EVAL_BATCH_SIZE", "65536"))
    grad_clip = float(os.environ.get("CTR_GRAD_CLIP", "0.0"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_cat_train, X_num_train, y_train = train_data
    X_cat_val, X_num_val, y_val = val_data
    x_cat_train = torch.from_numpy(X_cat_train).long()
    x_num_train = torch.from_numpy(X_num_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    x_cat_val = torch.from_numpy(X_cat_val).long()
    x_num_val = torch.from_numpy(X_num_val).float()
    train_dataset = torch.utils.data.TensorDataset(x_cat_train, x_num_train, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    if model_type == "LR":
        model = LRModel(cat_dims, num_dim)
    elif model_type == "FM":
        model = FMModel(cat_dims, num_dim, embed_dim)
    elif model_type == "DNN":
        model = DNNModel(cat_dims, num_dim, embed_dim, hidden_dims, dropout)
    elif model_type == "DCN":
        model = DCNModel(cat_dims, num_dim, embed_dim, hidden_dims, dropout, cross_layers)
    else:
        model = DeepFMModel(cat_dims, num_dim, embed_dim, hidden_dims, dropout)
    model.to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    step = 0
    model.train()
    while True:
        for batch in train_loader:
            if time.time() - t_start >= TIME_BUDGET:
                break
            x_cat_b, x_num_b, y_b = batch
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            x_num_b = x_num_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_cat_b, x_num_b).squeeze(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
        if time.time() - t_start >= TIME_BUDGET:
            break
    model.eval()
    with torch.no_grad():
        val_dataset = torch.utils.data.TensorDataset(x_cat_val, x_num_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False)
        probs_list = []
        for x_cat_b, x_num_b in val_loader:
            x_cat_b = x_cat_b.to(device, non_blocking=True)
            x_num_b = x_num_b.to(device, non_blocking=True)
            logits = model(x_cat_b, x_num_b).squeeze(-1)
            probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)
    auc = _auc_score(y_val, probs)
    logloss = _logloss(y_val, probs)
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0
    num_params = sum(p.numel() for p in model.parameters())
    print("---")
    print(f"val_auc:          {auc:.6f}")
    print(f"val_logloss:      {logloss:.6f}")
    print(f"training_seconds: {t_end - t_start:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
    print(f"model:            {model_type}")
    print(f"embed_dim:        {embed_dim}")
    print(f"hidden_dims:      {hidden_dims}")
    print(f"dropout:          {dropout}")
    print(f"cross_layers:     {cross_layers}")
    print(f"lr:               {lr}")
    print(f"batch_size:       {batch_size}")


if __name__ == "__main__":
    run_ctr()
