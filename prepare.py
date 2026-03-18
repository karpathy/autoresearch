import os
import json
import argparse
import zipfile

import numpy as np
import pandas as pd
import requests

CTR_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CTR_DATASET_DIR = os.path.join(CTR_CACHE_DIR, "datasets")
FEATURE_CONFIG_NAMES = ("features.json", "feature_config.json")

DATASET_URLS = {
    "Criteo_x1": "https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip?download=true",
    "Criteo_x2": "https://huggingface.co/datasets/reczoo/Criteo_x2/resolve/main/Criteo_x2.zip?download=true",
    "Criteo_x4": "https://huggingface.co/datasets/reczoo/Criteo_x4/resolve/main/Criteo_x4.zip?download=true",
    "Avazu_x1": "https://huggingface.co/datasets/reczoo/Avazu_x1/resolve/main/Avazu_x1.zip?download=true",
    "Avazu_x2": "https://huggingface.co/datasets/reczoo/Avazu_x2/resolve/main/Avazu_x2.zip?download=true",
    "Avazu_x4": "https://huggingface.co/datasets/reczoo/Avazu_x4/resolve/main/Avazu_x4.zip?download=true",
    "KKBox_x1": "https://huggingface.co/datasets/reczoo/KKBox_x1/resolve/main/KKBox_x1.zip?download=true",
    "Frappe_x1": "https://huggingface.co/datasets/reczoo/Frappe_x1/resolve/main/Frappe_x1.zip?download=true",
    "MovielensLatest_x1": "https://huggingface.co/datasets/reczoo/MovielensLatest_x1/resolve/main/MovielensLatest_x1.zip?download=true",
    "TaobaoAd_x1": "https://huggingface.co/datasets/reczoo/TaobaoAd_x1/resolve/main/TaobaoAd_x1.zip?download=true",
    "AmazonElectronics_x1": "https://huggingface.co/datasets/reczoo/AmazonElectronics_x1/resolve/main/AmazonElectronics_x1.zip?download=true",
    "iPinYou_x1": "https://huggingface.co/datasets/reczoo/iPinYou_x1/resolve/main/iPinYou_x1.zip?download=true",
    "MicroVideo1.7M_x1": "https://huggingface.co/datasets/reczoo/MicroVideo1.7M_x1/resolve/main/MicroVideo1.7M_x1.zip?download=true",
    "KuaiVideo_x1": "https://huggingface.co/datasets/reczoo/KuaiVideo_x1/resolve/main/KuaiVideo_x1.zip?download=true",
    "MIND_small_x1": "https://huggingface.co/datasets/reczoo/MIND_small_x1/resolve/main/MIND_small_x1.zip?download=true",
}


def _read_raw_data(path):
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _download_file(url, dst_path):
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dst_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _extract_zip(zip_path, target_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def _find_data_file(root_dir):
    preferred = [
        "train.parquet",
        "train.pq",
        "train.csv",
        "data.parquet",
        "data.csv",
    ]
    for name in preferred:
        candidate = os.path.join(root_dir, name)
        if os.path.isfile(candidate):
            return candidate
    candidates = []
    for base, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith((".parquet", ".pq", ".csv")):
                candidates.append(os.path.join(base, fname))
    if not candidates:
        raise FileNotFoundError(f"No data file found under {root_dir}")
    return sorted(candidates)[0]


def _download_dataset(dataset_id):
    url = DATASET_URLS.get(dataset_id)
    if not url:
        raise ValueError(f"Unknown dataset id: {dataset_id}")
    target_dir = os.path.join(CTR_DATASET_DIR, dataset_id)
    os.makedirs(target_dir, exist_ok=True)
    try:
        _find_data_file(target_dir)
        return target_dir
    except FileNotFoundError:
        pass
    zip_path = os.path.join(target_dir, f"{dataset_id}.zip")
    if not os.path.isfile(zip_path):
        _download_file(url, zip_path)
    _extract_zip(zip_path, target_dir)
    return target_dir


def _resolve_data_path(data_path):
    if os.path.isfile(data_path):
        return data_path
    if os.path.isdir(data_path):
        return _find_data_file(data_path)
    dataset_id = data_path.strip()
    if dataset_id in DATASET_URLS:
        dataset_dir = _download_dataset(dataset_id)
        return _find_data_file(dataset_dir)
    raise FileNotFoundError(f"Data path not found: {data_path}")


def _output_dir_for_data_path(data_path):
    parent = os.path.basename(os.path.dirname(data_path))
    stem = os.path.splitext(os.path.basename(data_path))[0]
    suffix = "sample" if stem.startswith("sample") else stem
    return os.path.join(CTR_CACHE_DIR, f"{parent}_{suffix}")


def _find_feature_config(data_path, feature_config_path):
    if feature_config_path:
        if os.path.isfile(feature_config_path):
            return feature_config_path
        raise FileNotFoundError(f"Feature config not found: {feature_config_path}")
    base_dir = os.path.dirname(data_path)
    for name in FEATURE_CONFIG_NAMES:
        candidate = os.path.join(base_dir, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Feature config not found under {base_dir}")


def _load_feature_config(data_path, feature_config_path):
    config_path = _find_feature_config(data_path, feature_config_path)
    with open(config_path, "r") as f:
        config = json.load(f)
    label_col = config.get("label_col", "label")
    cat_cols = list(config.get("categorical_cols") or [])
    num_cols = list(config.get("numerical_cols") or [])
    if not cat_cols or not num_cols:
        raise ValueError(f"Feature config missing categorical_cols or numerical_cols: {config_path}")
    return label_col, cat_cols, num_cols, config_path


def _resolve_columns(df, label_col, cat_cols, num_cols):
    if not cat_cols and not num_cols:
        for c in df.columns:
            if c == label_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                num_cols.append(c)
            else:
                cat_cols.append(c)
    else:
        used = set(cat_cols) | set(num_cols) | {label_col}
        for c in df.columns:
            if c in used:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                num_cols.append(c)
            else:
                cat_cols.append(c)
    return cat_cols, num_cols


def _split_df(df, test_ratio, seed):
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "val"].copy()
        if len(train_df) > 0 and len(val_df) > 0:
            return train_df, val_df
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(df) * test_ratio))
    val_df = df.iloc[:n_val].copy()
    train_df = df.iloc[n_val:].copy()
    return train_df, val_df


def _encode_categorical(series, categories):
    cat = pd.Categorical(series, categories=categories)
    codes = np.asarray(cat.codes)
    codes = codes + 1
    codes[codes < 0] = 0
    return codes


def prepare_ctr_data(
    data_path,
    label_col=None,
    cat_cols=None,
    num_cols=None,
    test_ratio=0.1,
    seed=42,
    force=False,
    feature_config_path=None,
):
    data_path = _resolve_data_path(data_path)
    output_dir = _output_dir_for_data_path(data_path)
    meta_path = os.path.join(output_dir, "meta.json")
    os.makedirs(output_dir, exist_ok=True)
    if not force and os.path.exists(meta_path):
        required = [
            os.path.join(output_dir, "X_cat_train.npy"),
            os.path.join(output_dir, "X_num_train.npy"),
            os.path.join(output_dir, "y_train.npy"),
            os.path.join(output_dir, "X_cat_val.npy"),
            os.path.join(output_dir, "X_num_val.npy"),
            os.path.join(output_dir, "y_val.npy"),
        ]
        if all(os.path.exists(p) for p in required):
            return
    cfg_label, cfg_cat_cols, cfg_num_cols, cfg_path = _load_feature_config(
        data_path, feature_config_path
    )
    if label_col is None:
        label_col = cfg_label
    cat_cols = list(cat_cols or cfg_cat_cols)
    num_cols = list(num_cols or cfg_num_cols)
    df = _read_raw_data(data_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")
    cat_cols, num_cols = _resolve_columns(df, label_col, cat_cols, num_cols)
    train_df, val_df = _split_df(df, test_ratio, seed)
    y_train = train_df[label_col].to_numpy().astype(np.float32)
    y_val = val_df[label_col].to_numpy().astype(np.float32)
    X_cat_train = np.zeros((len(train_df), len(cat_cols)), dtype=np.int64)
    X_cat_val = np.zeros((len(val_df), len(cat_cols)), dtype=np.int64)
    cat_dims = []
    for i, col in enumerate(cat_cols):
        categories = train_df[col].astype("category").cat.categories
        X_cat_train[:, i] = _encode_categorical(train_df[col], categories)
        X_cat_val[:, i] = _encode_categorical(val_df[col], categories)
        cat_dims.append(int(len(categories) + 1))
    X_num_train = np.zeros((len(train_df), len(num_cols)), dtype=np.float32)
    X_num_val = np.zeros((len(val_df), len(num_cols)), dtype=np.float32)
    num_stats = []
    for i, col in enumerate(num_cols):
        tr = train_df[col].astype(np.float32)
        mean = float(np.nanmean(tr))
        std = float(np.nanstd(tr))
        if std == 0.0:
            std = 1.0
        num_stats.append([mean, std])
        tr_filled = np.nan_to_num(tr, nan=mean)
        val_filled = np.nan_to_num(val_df[col].astype(np.float32), nan=mean)
        X_num_train[:, i] = (tr_filled - mean) / std
        X_num_val[:, i] = (val_filled - mean) / std
    meta = {
        "data_path": data_path,
        "label_col": label_col,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_dims": cat_dims,
        "num_stats": num_stats,
        "test_ratio": test_ratio,
        "seed": seed,
        "feature_config": cfg_path,
        "output_dir": output_dir,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    np.save(os.path.join(output_dir, "X_cat_train.npy"), X_cat_train)
    np.save(os.path.join(output_dir, "X_num_train.npy"), X_num_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_cat_val.npy"), X_cat_val)
    np.save(os.path.join(output_dir, "X_num_val.npy"), X_num_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)


def load_ctr_data(data_path, feature_config_path=None):
    data_path = _resolve_data_path(data_path)
    output_dir = _output_dir_for_data_path(data_path)
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    X_cat_train = np.load(os.path.join(output_dir, "X_cat_train.npy"))
    X_num_train = np.load(os.path.join(output_dir, "X_num_train.npy"))
    y_train = np.load(os.path.join(output_dir, "y_train.npy"))
    X_cat_val = np.load(os.path.join(output_dir, "X_cat_val.npy"))
    X_num_val = np.load(os.path.join(output_dir, "X_num_val.npy"))
    y_val = np.load(os.path.join(output_dir, "y_val.npy"))
    return meta, (X_cat_train, X_num_train, y_train), (X_cat_val, X_num_val, y_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for CTR autoresearch")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--feature-config", default="")
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_ctr_data(
        data_path=args.data_path,
        test_ratio=args.test_ratio,
        seed=args.seed,
        force=args.force,
        feature_config_path=args.feature_config or None,
    )
    print("Done! Ready to train.")
