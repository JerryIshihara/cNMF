"""Batched PyTorch multiplicative-update NMF solver."""

import numpy as np

from . import utils

# ---------------------------------------------------------------------
# MU solver (batch-aware; R>=1 replicates per launch)
# ---------------------------------------------------------------------


def _validate_mu_runtime(nmf_kwargs):
    """Reject sklearn options that this Frobenius-only MU kernel cannot honor."""
    legacy_regularization = {"alpha", "regularization"}.intersection(nmf_kwargs)
    if legacy_regularization:
        raise ValueError(
            "GPU solver='mu' does not accept deprecated alpha/regularization; "
            "use alpha_W and alpha_H"
        )

    beta_loss = nmf_kwargs.get("beta_loss", "frobenius")
    if not (beta_loss == 2 or str(beta_loss).lower() == "frobenius"):
        raise ValueError("GPU solver='mu' supports only beta_loss='frobenius'")

    alpha_w = float(nmf_kwargs.get("alpha_W", 0.0))
    alpha_h_raw = nmf_kwargs.get("alpha_H", "same")
    alpha_h = alpha_w if alpha_h_raw == "same" else float(alpha_h_raw)
    if alpha_w != 0.0 or alpha_h != 0.0:
        raise ValueError(
            "GPU solver='mu' does not yet support alpha_W/alpha_H regularization"
        )


def _mu_step(W, H, Xg, eps):
    """One MU update, sklearn order: update W from old H, then H from new W.

    Accepts either 2D single-replicate tensors or stacked `[R,...]` replicate
    tensors with shared `Xg[1,n,g]`. Operations are out-of-place for compile.
    """
    Ht = H.transpose(-2, -1)                               # [g,k] or [R,g,k]
    denominator = W @ (H @ Ht)
    denominator = denominator.where(denominator != 0, eps)
    W = W * ((Xg @ Ht) / denominator)                      # W *= XHᵀ / (W·HHᵀ)   (uses old H)
    Wt = W.transpose(-2, -1)                               # [k,n] or [R,k,n]
    denominator = (Wt @ W) @ H
    denominator = denominator.where(denominator != 0, eps)
    H = H * ((Wt @ Xg) / denominator)                      # H *= WᵀX / (WᵀW·H)   (uses new W)
    return W, H


def _mu_step_fixed_h(W, H, Xg, eps):
    """One fixed-H MU update; only W changes. Supports 2D or stacked W."""
    Ht = H.transpose(-2, -1)                               # [g,k] or [1,g,k]
    denominator = W @ (H @ Ht)
    denominator = denominator.where(denominator != 0, eps)
    return W * ((Xg @ Ht) / denominator)


def _fit_mu(torch, Xg, W, H, eps, max_iter, tol, step, block, tf32, device):
    """Run full MU until `max_iter` or all replicate slices meet `tol`."""
    utils._check_runtime_tensors(Xg, W, H, eps)
    xnorm2 = utils._sq_norm(Xg)                            # ‖X‖² once; error check avoids [R,n,g]
    err_init = prev_err = None
    with torch.no_grad(), utils._cuda_tf32(torch, tf32, device):
        it = 0
        while it < max_iter:
            n = min(block, max_iter - it)
            for _ in range(n):                             # MU updates run inside the (compiled) step
                W, H = step(W, H, Xg, eps)
            it += n
            err = utils._recon_err(Xg, W, H, xnorm2)
            if err_init is None:
                err_init = err.clamp_min(1e-30)            # avoid 0/0 on a degenerate (all-zero) slice
            elif prev_err is not None and bool((((prev_err - err) / err_init) < tol).all()):
                break
            prev_err = err
    return W, H


def _fit_mu_fixed_h(torch, Xg, W, H, eps, max_iter, tol, step, block, tf32, device):
    """Run fixed-H MU until `max_iter` or all replicate slices meet `tol`."""
    utils._check_runtime_tensors(Xg, W, H, eps)
    xnorm2 = utils._sq_norm(Xg)                            # ‖X‖² once; error check avoids [R,n,g]
    err_init = prev_err = None
    with torch.no_grad(), utils._cuda_tf32(torch, tf32, device):
        it = 0
        while it < max_iter:
            n = min(block, max_iter - it)
            for _ in range(n):
                W = step(W, H, Xg, eps)
            it += n
            err = utils._recon_err(Xg, W, H, xnorm2)
            if err_init is None:
                err_init = err.clamp_min(1e-30)            # avoid 0/0 on a degenerate (all-zero) slice
            elif prev_err is not None and bool((((prev_err - err) / err_init) < tol).all()):
                break
            prev_err = err
    return W


def _nmf_gpu_mu(X, seeds, nmf_kwargs, gpu_kwargs=None):
    """Run full or fixed-H same-k MU replicates; return one `(H, W)` per seed."""
    torch = utils._loud_import_torch()

    seeds = utils._normalize_seeds(seeds)

    rc = utils._gpu_setup(torch, X, nmf_kwargs, gpu_kwargs)
    _validate_mu_runtime(nmf_kwargs)
    init = nmf_kwargs.get("init")
    update_h = nmf_kwargs.get("update_H", True) is not False
    fixed_h = None if update_h else utils._to_checked_fixed_h(
        nmf_kwargs.get("H"), rc.k, rc.Xnp.shape[1]
    )

    # TODO: stream sparse/row-blocked X instead of requiring full dense X in RAM/VRAM.
    Xb = torch.as_tensor(rc.Xnp, dtype=rc.dtype, device=rc.device).unsqueeze(0)      # [1, n, g] shared

    Ws, Hs = [], []
    for s in seeds:
        W0, H0 = utils._init_wh(rc.Xnp, rc.k, s, init)                               # (usages, spectra)
        Ws.append(np.ascontiguousarray(W0))
        if update_h:
            Hs.append(np.ascontiguousarray(H0))
    W = torch.as_tensor(np.stack(Ws, 0), dtype=rc.dtype, device=rc.device)          # [R, n, k]
    if update_h:
        H = torch.as_tensor(np.stack(Hs, 0), dtype=rc.dtype, device=rc.device)      # [R, k, g]
        step, block = utils._execution_plan(torch, rc.opt, rc.device, _mu_step)
        W, H = _fit_mu(
            torch, Xb, W, H, rc.eps, rc.max_iter, rc.tol,
            step, block, utils._want_tf32(torch, rc), rc.device,
        )
    else:
        H = torch.as_tensor(
            np.ascontiguousarray(fixed_h), dtype=rc.dtype, device=rc.device
        ).unsqueeze(0)                                                              # [1, k, g] shared
        step, block = utils._execution_plan(
            torch, rc.opt, rc.device, _mu_step_fixed_h
        )
        W = _fit_mu_fixed_h(
            torch, Xb, W, H, rc.eps, rc.max_iter, rc.tol,
            step, block, utils._want_tf32(torch, rc), rc.device,
        )

    Wc = W.cpu().double().numpy()
    Hc = H.cpu().double().numpy()
    if update_h:
        return [(Hc[r], Wc[r]) for r in range(len(seeds))]
    return [(Hc[0], Wc[r]) for r in range(len(seeds))]
