from __future__ import annotations

import torch

from .compile import maybe_compile_function


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor, step_t: torch.Tensor, lr_t: torch.Tensor, beta1_t: torch.Tensor, beta2_t: torch.Tensor, eps_t: torch.Tensor, wd_t: torch.Tensor):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(
    stacked_grads: torch.Tensor,
    stacked_params: torch.Tensor,
    momentum_buffer: torch.Tensor,
    second_momentum_buffer: torch.Tensor,
    momentum_t: torch.Tensor,
    lr_t: torch.Tensor,
    wd_t: torch.Tensor,
    beta2_t: torch.Tensor,
    ns_steps: int,
    red_dim: int,
):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Muon's Newton-Schulz update runs on a normalized bf16 copy to keep the
    # fused kernel cheap while still approximating the matrix orthogonalization.
    x = g.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_mat = x.mT @ x
            b_mat = b * a_mat + c * (a_mat @ a_mat)
            x = a * x + x @ b_mat
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_mat = x @ x.mT
            b_mat = b * a_mat + c * (a_mat @ a_mat)
            x = a * x + b_mat @ x
    g = x

    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    # Only decay coordinates that agree with the update direction; this matches
    # the reference Muon implementation's sign-aware decay.
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups, *, optimizer_compile_backend: str, compile_mode: str):
        super().__init__(param_groups, defaults={})
        self._adamw_step_fn = maybe_compile_function(
            adamw_step_fused,
            backend=optimizer_compile_backend,
            compile_mode=compile_mode,
            dynamic=False,
            fullgraph=True,
        )
        self._muon_step_fn = maybe_compile_function(
            muon_step_fused,
            backend=optimizer_compile_backend,
            compile_mode=compile_mode,
            dynamic=False,
            fullgraph=True,
        )

        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            grad = p.grad if p.grad.dtype == torch.float32 else p.grad.float()
            self._adamw_step_fn(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device = p.shape, p.device
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=torch.float32, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=device)
        active_indices = [idx for idx, param in enumerate(params) if param.grad is not None]
        if not active_indices:
            return
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        # Grouping same-shaped matrices lets Muon update them as one compiled
        # stack, which is much friendlier to torch.compile than per-parameter Python loops.
        active_params = [params[idx] for idx in active_indices]
        stacked_grads = torch.stack([param.grad.float() for param in active_params])
        stacked_params = torch.stack(active_params)
        scatter_state_back = len(active_params) != len(params)
        if scatter_state_back:
            index_tensor = torch.tensor(active_indices, device=device, dtype=torch.long)
            momentum_buffer = state["momentum_buffer"].index_select(0, index_tensor).clone()
            second_momentum_buffer = state["second_momentum_buffer"].index_select(0, index_tensor).clone()
        else:
            index_tensor = None
            momentum_buffer = state["momentum_buffer"]
            second_momentum_buffer = state["second_momentum_buffer"]
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self._muon_step_fn(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        if scatter_state_back:
            state["momentum_buffer"].index_copy_(0, index_tensor, momentum_buffer)
            state["second_momentum_buffer"].index_copy_(0, index_tensor, second_momentum_buffer)
        torch._foreach_copy_(active_params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
