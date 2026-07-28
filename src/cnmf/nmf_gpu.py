#!/usr/bin/env python
"""PyTorch Frobenius-MU NMF backend for cNMF.

The kernel is cNMF-compatible: it returns `(spectra, usages) = (H, W)` as numpy
float64, while compute dtype/device are controlled by `gpu_kwargs`.

Supported runtime options:
    device: auto|cuda|cuda:N|mps|cpu     auto = CUDA -> MPS -> CPU
    dtype:  auto|fp32|fp64|bf16          auto = fp64 on CPU, fp32 on GPU
    allow_tf32, compile, eps, check_every, compile_block, batch

Explicit unavailable GPUs raise instead of falling back to CPU. MPS is fp32-only
for this kernel; bf16 is explicit CUDA-only storage and matmul. The implementation
also supports batched same-k replicates for factorize and fixed-H consensus refits.
"""
import contextlib
import functools
from collections import namedtuple

import numpy as np


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------


_TRUTHY = {"1", "true", "yes", "on"}


DEFAULT_NMF = {
    "max_iter": 1000,
    "tol": 1e-4,
    "init": "random",
}


_SKLEARN_EPSILON = float(np.finfo(np.float32).eps)


DEFAULT_GPU = {
    "device": "auto",
    "dtype": "auto",
    "allow_tf32": False,
    "compile": False,
    "eps": _SKLEARN_EPSILON,
    "check_every": 10,
    "compile_block": 1,
    "batch": 1,             # factorize replicates per launch
}


# ---------------------------------------------------------------------
# CLI argument parsing and option resolution
# ---------------------------------------------------------------------


def parse_gpu_args(parser):
    """Register cNMF CLI flags for the optional PyTorch GPU NMF engine."""
    group = parser.add_argument_group("NMF engine options")
    group.add_argument("--engine", type=str.lower, choices=["cpu", "gpu"], help="[factorize,consensus] NMF engine to use (default cpu)")
    group.add_argument("--gpu-device", type=str, help="[factorize,consensus,gpu] Device for GPU NMF: auto, cpu, cuda, cuda:N, or mps")
    group.add_argument("--gpu-dtype", type=str.lower, choices=["auto", "fp32", "fp64", "bf16"], help="[factorize,consensus,gpu] Storage and matmul dtype for GPU NMF (default auto)")
    group.add_argument("--gpu-allow-tf32", action="store_const", const=True, help="[factorize,consensus,gpu] Allow TF32 for CUDA fp32 matrix multiplication")
    group.add_argument("--gpu-compile", action="store_const", const=True, help="[factorize,consensus,gpu] Enable torch.compile for the GPU NMF update step")
    group.add_argument("--gpu-eps", type=float, help="[factorize,consensus,gpu] Replacement for exactly-zero MU denominators")
    group.add_argument("--gpu-check-every", type=int, help="[factorize,consensus,gpu] Eager-mode convergence check interval")
    group.add_argument("--gpu-compile-block", type=int, help="[factorize,consensus,gpu] Number of MU iterations per compiled block")
    group.add_argument("--gpu-batch", type=int, help="[factorize] Replicates run per GPU launch (batched MU); 1 = single-replicate")
    return parser


def gpu_kwargs_from_args(args):
    """Collect parsed cNMF CLI GPU flags into a kernel gpu_kwargs dict."""
    raw = {
        "device": args.gpu_device,
        "dtype": args.gpu_dtype,
        "allow_tf32": args.gpu_allow_tf32,
        "compile": args.gpu_compile,
        "eps": args.gpu_eps,
        "check_every": args.gpu_check_every,
        "compile_block": args.gpu_compile_block,
        "batch": args.gpu_batch,
    }
    if args.engine != "gpu":
        if any(value is not None for value in raw.values()):
            raise ValueError("GPU options require --engine gpu")
        return None
    return _resolve_gpu_opts(raw)


def validate_engine_args_for_command(args, available_commands):
    """Engine/GPU CLI options are only valid for commands that support the selected engine."""
    available_commands = tuple(available_commands)
    if args.command in available_commands:
        return

    gpu_arg_names = [
        "engine",
        "gpu_device",
        "gpu_dtype",
        "gpu_allow_tf32",
        "gpu_compile",
        "gpu_eps",
        "gpu_check_every",
        "gpu_compile_block",
        "gpu_batch",
    ]
    if any(getattr(args, name) is not None for name in gpu_arg_names):
        commands = ", ".join(available_commands)
        raise ValueError(f"NMF engine/GPU options are only valid with: {commands}")


def _resolve_gpu_opts(gpu_kwargs):
    """Merge Nextflow-provided gpu_kwargs over defaults into a typed opts dict."""
    raw = dict(gpu_kwargs or {})

    def parse_bool(value, default):
        return default if value is None else str(value).strip().lower() in _TRUTHY

    def parse_typed(value, default, cast, normalize=None):
        parsed = default if value is None else cast(value)
        return normalize(parsed) if normalize is not None else parsed

    def parse_positive_int(value, default):
        return max(1, parse_typed(value, default, int))

    return dict(
        device        = parse_typed(raw.get("device"), DEFAULT_GPU["device"], str, str.lower),
        dtype         = parse_typed(raw.get("dtype"), DEFAULT_GPU["dtype"], str, str.lower),
        allow_tf32    = parse_bool(raw.get("allow_tf32"), DEFAULT_GPU["allow_tf32"]),
        compile       = parse_bool(raw.get("compile"), DEFAULT_GPU["compile"]),
        eps           = parse_typed(raw.get("eps"), DEFAULT_GPU["eps"], float),
        check_every   = parse_positive_int(raw.get("check_every"), DEFAULT_GPU["check_every"]),
        compile_block = parse_positive_int(raw.get("compile_block"), DEFAULT_GPU["compile_block"]),
        batch         = parse_positive_int(raw.get("batch"), DEFAULT_GPU["batch"]),
    )


# ---------------------------------------------------------------------
# Runtime backend selection (device / dtype / TF32)
# ---------------------------------------------------------------------


def _select_device(torch, requested):
    """Resolve device, raising for explicit unavailable CUDA/MPS requests."""
    gpu_availability = {
        "cuda": torch.cuda.is_available,
        "mps": torch.backends.mps.is_available,
    }
    valid_bases = {"cpu", *gpu_availability}

    if requested == "auto":
        for candidate, is_available in gpu_availability.items():
            if is_available():
                return candidate
        return "cpu"

    base = requested.split(":")[0]
    if base not in valid_bases:
        raise ValueError(f"device={requested!r} not recognized; use auto|cpu|cuda|cuda:N|mps.")

    if base in gpu_availability and not gpu_availability[base]():
        raise RuntimeError(f"device={requested!r} requested but {base.upper()} is unavailable "
                           "(use device='auto' or 'cpu').")
    return requested


def _select_storage(torch, requested, device):
    """Resolve storage/matmul dtype for the selected device."""
    base = device.split(":")[0]
    dtype_map = {
        "fp32": torch.float32,
        "fp64": torch.float64,
        "bf16": torch.bfloat16,
    }

    if requested == "auto":
        requested = "fp64" if base == "cpu" else "fp32"

    if requested not in dtype_map:
        choices = "|".join(["auto", *dtype_map])
        raise ValueError(f"dtype={requested!r} not recognized; use {choices}.")

    if requested == "fp64" and base == "mps":
        raise RuntimeError("dtype='fp64' requested but MPS has no fp64 "
                           "(use dtype='auto'/'fp32', or device='cpu'/'cuda').")

    if requested == "bf16":
        if base != "cuda":
            raise RuntimeError("dtype='bf16' is only supported on CUDA in this kernel "
                               "(use dtype='auto'/'fp32' or device='cuda').")
        is_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(is_supported) and not is_supported():
            raise RuntimeError("dtype='bf16' requested but this CUDA device does not support bf16.")

    return dtype_map[requested]


@contextlib.contextmanager
def _cuda_tf32(torch, enable, device):
    """Temporarily set CUDA TF32 matmul flags; no-op off CUDA."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        yield
        return
    prev_allow = torch.backends.cuda.matmul.allow_tf32
    prev_prec = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.set_float32_matmul_precision("high" if enable else "highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow
        torch.set_float32_matmul_precision(prev_prec)


# ---------------------------------------------------------------------
# Input validation, lazy imports, and initialization
# ---------------------------------------------------------------------


def _to_checked_array(X):
    """Materialize X dense and enforce the NMF preconditions: finite and non-negative."""
    # TODO: sparse RAM path. Sparse X is still densified by this prototype; add a dense-size
    # preflight and/or row-blocked sparse loading before materializing full X in host RAM.
    Xnp = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if Xnp.ndim != 2:
        raise ValueError("NMF input X must be a 2D matrix")
    if 0 in Xnp.shape:
        raise ValueError("NMF input X must have at least one row and one column")
    if not np.isfinite(Xnp).all():
        raise ValueError("NMF input X contains NaN/inf")
    if Xnp.size and Xnp.min() < 0:
        raise ValueError(f"NMF requires non-negative input X; found min(X) = {float(Xnp.min()):.4g}")
    return np.ascontiguousarray(Xnp)


def _to_checked_fixed_h(H, k, n_features):
    """Materialize and validate fixed H for update_H=False consensus refits."""
    if H is None:
        raise ValueError("update_H=False requires a fixed H matrix")
    Hnp = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
    if Hnp.ndim != 2:
        raise ValueError("fixed H must be a 2D matrix")
    if Hnp.shape != (k, n_features):
        raise ValueError(f"fixed H shape must be ({k}, {n_features}); got {Hnp.shape}")
    if not np.isfinite(Hnp).all():
        raise ValueError("fixed H contains NaN/inf")
    if Hnp.size and Hnp.min() < 0:
        raise ValueError(f"NMF requires non-negative fixed H; found min(H) = {float(Hnp.min()):.4g}")
    return Hnp


def _loud_import_torch():
    """Import torch with an actionable environment error for pipeline users."""
    try:
        import torch
    except ModuleNotFoundError as e:
        if e.name != "torch":
            raise
        raise RuntimeError(
            "PyTorch is required for GPU NMF but is not installed in the active environment. "
            "Install cNMF with GPU support in that environment using "
            "`python -m pip install -e \".[gpu]\"`, or install a CUDA-compatible torch build "
            "before installing the cNMF GPU extra."
        ) from e
    return torch


def _loud_import_initialize_nmf():
    """Import sklearn's NMF initializer with an actionable environment/version error."""
    try:
        from sklearn.decomposition._nmf import _initialize_nmf
    except ModuleNotFoundError as e:
        if e.name != "sklearn":
            raise
        raise RuntimeError(
            "scikit-learn is required for sklearn-compatible NMF initialization but is not "
            "installed in the active environment. Install cNMF into that environment using "
            "`python -m pip install -e .` or `python -m pip install -e \".[gpu]\"`."
        ) from e
    except ImportError as e:
        raise RuntimeError(
            "The installed scikit-learn does not expose sklearn.decomposition._nmf._initialize_nmf. "
            "Use a supported scikit-learn version or update the GPU NMF initializer adapter for "
            "this scikit-learn version."
        ) from e
    return _initialize_nmf


def _init_wh(Xnp, k, seed, init):
    """Initialize W/H with sklearn parity; `init=None` uses DEFAULT_NMF."""
    if init == "custom":
        raise NotImplementedError("nmf_gpu does not support init='custom'")
    _initialize_nmf = _loud_import_initialize_nmf()
    return _initialize_nmf(
        Xnp,
        n_components=k,
        init=(init or DEFAULT_NMF["init"]),
        random_state=seed,
    )


# ---------------------------------------------------------------------
# Multiplicative-update steppers (one MU iteration; batch-aware)
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# MU fit loops (iterate to convergence; batch-aware)
# ---------------------------------------------------------------------


def _recon_err(Xg, W, H, xnorm2):
    """Per-replicate ‖X − WH‖_F via the identity ‖X‖² − 2⟨X,WH⟩ + ‖WH‖², using only small
    [R,k,g]/[R,k,k] matmuls (WᵀX, WᵀW, HHᵀ) — it NEVER materializes the [R,n,g] product, so batched
    fits scale to millions of cells (a full [R,n,g] residual would be ~R·n·g floats, tens of GB).
    Returns a 0-dim tensor (unbatched) or shape-[R] (batched); xnorm2 = ‖X‖² is precomputed once."""
    Wt = W.transpose(-2, -1)
    cross = ((Wt @ Xg) * H).sum(dim=(-2, -1))                          # ⟨X, WH⟩ per replicate
    whnorm = ((Wt @ W) * (H @ H.transpose(-2, -1))).sum(dim=(-2, -1))  # ‖WH‖² per replicate
    return (xnorm2 - 2.0 * cross + whnorm).clamp_min(0).sqrt()


def _sq_norm(X):
    """‖X‖² (every element squared, then summed) in row chunks. torch.dot is BLAS-backed and caps
    its vector length at 2³¹ elements, so a flat dot overflows past ~2.1B entries (e.g. 2.4M cells
    × 2000 genes = 4.8B); this chunked reduction is index-safe at any size and also bounds the
    squaring temporary. Returns a 0-dim tensor."""
    rows = X.shape[0]
    step = max(1, (1 << 28) // max(1, X.numel() // max(1, rows)))      # ~2²⁸ elems/chunk (~1GB fp32)
    total = X.new_zeros(())
    for i in range(0, rows, step):
        total = total + X[i:i + step].square().sum()
    return total


def _fit_mu(torch, Xg, W, H, eps, max_iter, tol, step, block, tf32, device):
    """Run full MU until `max_iter` or all replicate slices meet `tol`."""
    _check_runtime_tensors(Xg, W, H, eps)
    xnorm2 = _sq_norm(Xg)                                  # ‖X‖² once; error check avoids [R,n,g]
    err_init = prev_err = None
    with torch.no_grad(), _cuda_tf32(torch, tf32, device):
        it = 0
        while it < max_iter:
            n = min(block, max_iter - it)
            for _ in range(n):                             # MU updates run inside the (compiled) step
                W, H = step(W, H, Xg, eps)
            it += n
            err = _recon_err(Xg, W, H, xnorm2)
            if err_init is None:
                err_init = err.clamp_min(1e-30)            # avoid 0/0 on a degenerate (all-zero) slice
            elif prev_err is not None and bool((((prev_err - err) / err_init) < tol).all()):
                break
            prev_err = err
    return W, H


def _fit_mu_fixed_h(torch, Xg, W, H, eps, max_iter, tol, step, block, tf32, device):
    """Run fixed-H MU until `max_iter` or all replicate slices meet `tol`."""
    _check_runtime_tensors(Xg, W, H, eps)
    xnorm2 = _sq_norm(Xg)                                  # ‖X‖² once; error check avoids [R,n,g]
    err_init = prev_err = None
    with torch.no_grad(), _cuda_tf32(torch, tf32, device):
        it = 0
        while it < max_iter:
            n = min(block, max_iter - it)
            for _ in range(n):
                W = step(W, H, Xg, eps)
            it += n
            err = _recon_err(Xg, W, H, xnorm2)
            if err_init is None:
                err_init = err.clamp_min(1e-30)            # avoid 0/0 on a degenerate (all-zero) slice
            elif prev_err is not None and bool((((prev_err - err) / err_init) < tol).all()):
                break
            prev_err = err
    return W


# ---------------------------------------------------------------------
# Execution plan, run context, and device-tensor helpers
# ---------------------------------------------------------------------


def _execution_plan(torch, opt, device, step_fn=_mu_step):
    """Return `(step_fn, block_size)` for eager or compiled execution."""
    use_compile = opt["compile"] and not device.startswith("mps")
    if use_compile:
        return torch.compile(step_fn), opt["compile_block"]
    return step_fn, opt["check_every"]


# Runtime context resolved once per factorize call.
_GpuRun = namedtuple("_GpuRun", "opt device dtype k max_iter tol eps Xnp")


def _gpu_setup(torch, X, nmf_kwargs, gpu_kwargs):
    """Validate common inputs and resolve device, dtype, eps, k, max_iter, and tol."""
    opt = _resolve_gpu_opts(gpu_kwargs)
    device = _select_device(torch, opt["device"])
    dtype = _select_storage(torch, opt["dtype"], device)
    k = int(nmf_kwargs["n_components"])
    if k < 1:
        raise ValueError("n_components must be >= 1")
    max_iter = int(nmf_kwargs.get("max_iter", DEFAULT_NMF["max_iter"]))
    tol = float(nmf_kwargs.get("tol", DEFAULT_NMF["tol"]))
    eps = _to_device_eps(torch, opt["eps"], dtype, device)
    Xnp = _to_checked_array(X)
    return _GpuRun(opt, device, dtype, k, max_iter, tol, eps, Xnp)


def _want_tf32(torch, rc):
    """TF32 applies only to explicitly-allowed CUDA fp32 matmul."""
    return rc.device.startswith("cuda") and rc.opt["allow_tf32"] and rc.dtype is torch.float32


def _to_device_factors(torch, W0, H0, dtype, device):
    """Move initialized factors to runtime dtype/device."""
    W = torch.as_tensor(np.ascontiguousarray(W0), dtype=dtype, device=device)        # usages  (cells x k)
    H = torch.as_tensor(np.ascontiguousarray(H0), dtype=dtype, device=device)        # spectra (k x genes)
    return W, H


def _to_device_eps(torch, eps, dtype, device):
    """Create the exact-zero denominator replacement on runtime dtype/device."""
    return torch.tensor(eps, dtype=dtype, device=device)


def _to_nmf_output(H, W):
    """Return spectra/usages as numpy float64 for compatibility; compute precision is unchanged."""
    return H.cpu().double().numpy(), W.cpu().double().numpy()


def _check_runtime_tensors(Xg, W, H, eps):
    """Require one dtype across X/W/H/eps."""
    dtypes = {Xg.dtype, W.dtype, H.dtype, eps.dtype}
    if len(dtypes) != 1:
        raise RuntimeError(f"NMF runtime tensors must share dtype; got {sorted(map(str, dtypes))}")


# ---------------------------------------------------------------------
# NMF kernels (batch-aware; R>=1 replicates per launch, R=1 = single instance)
# ---------------------------------------------------------------------


def _nmf_gpu_mu(X, seeds, nmf_kwargs, gpu_kwargs=None):
    """Run same-X, same-k MU replicates in one launch; return one `(H, W)` per seed."""
    torch = _loud_import_torch()

    seeds = [None if s is None else int(s) for s in seeds]
    if not seeds:
        raise ValueError("seeds must be a non-empty list of per-replicate random states")

    rc = _gpu_setup(torch, X, nmf_kwargs, gpu_kwargs)
    init = nmf_kwargs.get("init")
    # TODO: stream sparse/row-blocked X instead of requiring full dense X in RAM/VRAM.
    Xb = torch.as_tensor(rc.Xnp, dtype=rc.dtype, device=rc.device).unsqueeze(0)      # [1, n, g] shared

    # Initialize each replicate independently, then stack.
    Ws, Hs = [], []
    for s in seeds:
        W0, H0 = _init_wh(rc.Xnp, rc.k, s, init)                                     # (usages, spectra)
        Ws.append(np.ascontiguousarray(W0)); Hs.append(np.ascontiguousarray(H0))
    W = torch.as_tensor(np.stack(Ws, 0), dtype=rc.dtype, device=rc.device)          # [R, n, k]
    H = torch.as_tensor(np.stack(Hs, 0), dtype=rc.dtype, device=rc.device)          # [R, k, g]

    step, block = _execution_plan(torch, rc.opt, rc.device)
    W, H = _fit_mu(torch, Xb, W, H, rc.eps, rc.max_iter, rc.tol, step, block, _want_tf32(torch, rc), rc.device)

    Hc = H.cpu().double().numpy(); Wc = W.cpu().double().numpy()
    return [(Hc[r], Wc[r]) for r in range(len(seeds))]


def _nmf_gpu_fixed_h(X, seeds, nmf_kwargs, gpu_kwargs=None):
    """Run fixed-H consensus refits in one launch; return one `(H, W)` per seed."""
    torch = _loud_import_torch()

    seeds = [None if s is None else int(s) for s in seeds]
    if not seeds:
        raise ValueError("seeds must be a non-empty list of per-replicate random states")

    rc = _gpu_setup(torch, X, nmf_kwargs, gpu_kwargs)
    init = nmf_kwargs.get("init")
    Hnp = _to_checked_fixed_h(nmf_kwargs.get("H"), rc.k, rc.Xnp.shape[1])             # fixed spectra [k, g]
    Xb = torch.as_tensor(rc.Xnp, dtype=rc.dtype, device=rc.device).unsqueeze(0)      # [1, n, g] shared

    # Initialize W independently per seed; share fixed H across slices.
    Ws = []
    for s in seeds:
        W0, _ = _init_wh(rc.Xnp, rc.k, s, init)
        Ws.append(np.ascontiguousarray(W0))
    W = torch.as_tensor(np.stack(Ws, 0), dtype=rc.dtype, device=rc.device)           # [R, n, k]
    H = torch.as_tensor(np.ascontiguousarray(Hnp), dtype=rc.dtype, device=rc.device).unsqueeze(0)  # [1, k, g] fixed

    step, block = _execution_plan(torch, rc.opt, rc.device, _mu_step_fixed_h)
    W = _fit_mu_fixed_h(torch, Xb, W, H, rc.eps, rc.max_iter, rc.tol, step, block, _want_tf32(torch, rc), rc.device)

    Hc = H.squeeze(0).cpu().double().numpy()                                         # shared fixed spectra [k, g]
    Wc = W.cpu().double().numpy()
    return [(Hc, Wc[r]) for r in range(len(seeds))]


def _nmf_gpu(X, nmf_kwargs, gpu_kwargs=None):
    """Single-replicate NMF API; dispatches to full MU or fixed-H refit."""
    kernel = _nmf_gpu_fixed_h if nmf_kwargs.get("update_H", True) is False else _nmf_gpu_mu
    (result,) = kernel(X, [nmf_kwargs.get("random_state")], nmf_kwargs, gpu_kwargs)
    return result


# ---------------------------------------------------------------------
# cNMF integration: engine wiring and adapters
# ---------------------------------------------------------------------


def configure_nmf_engine(cnmf_obj, engine="cpu", gpu_kwargs=None):
    """Install GPU `_nmf` and batched factorize hooks on a cNMF instance."""
    if engine not in ("cpu", "gpu"):
        raise ValueError("engine must be 'cpu' or 'gpu'")
    if engine == "cpu":
        return cnmf_obj

    def _gpu_nmf(X, nmf_kwargs):
        nmf_kwargs = dict(nmf_kwargs)
        nmf_kwargs["engine"] = "gpu"
        nmf_kwargs["gpu"] = gpu_kwargs or {}
        return nmf_gpu(cnmf_obj, X, nmf_kwargs)

    cnmf_obj._nmf = _gpu_nmf

    # Factorize packs same-k replicates according to gpu_kwargs["batch"].
    cnmf_obj.factorize = functools.partial(factorize_gpu, cnmf_obj, gpu_kwargs or {})
    return cnmf_obj


def nmf_gpu(self, X, nmf_kwargs):
    """cNMF `_nmf` adapter; `self` is ignored for monkeypatch compatibility."""
    nmf_kwargs = dict(nmf_kwargs)
    gpu_kwargs = nmf_kwargs.pop("gpu", None)
    nmf_kwargs.pop("engine", None)
    return _nmf_gpu(X, nmf_kwargs, gpu_kwargs)


def factorize_gpu(cnmf_obj, gpu_kwargs, worker_i=0, total_workers=1, skip_completed_runs=False):
    """GPU `factorize` drop-in: group worker jobs by k, batch seeds, write iter spectra."""
    import scanpy as sc
    import yaml
    import pandas as pd
    from collections import OrderedDict
    from .cnmf import load_df_from_npz, save_df_to_npz, worker_filter

    batch = _resolve_gpu_opts(gpu_kwargs)["batch"]

    run_params  = load_df_from_npz(cnmf_obj.paths['nmf_replicate_parameters'])
    norm_counts = sc.read(cnmf_obj.paths['normalized_counts'])
    base_kwargs = yaml.load(open(cnmf_obj.paths['nmf_run_parameters']), Loader=yaml.FullLoader)

    if not skip_completed_runs:
        job_idx = worker_filter(range(len(run_params)), worker_i, total_workers)
    else:
        job_idx = worker_filter(run_params.index[run_params['completed'] == False], worker_i, total_workers)

    genes = norm_counts.var.index
    # Densify X once. _nmf_gpu_mu -> _gpu_setup calls X.toarray(), so passing the sparse matrix
    # would re-densify the full n×g matrix on *every* batch (O(#batches) redundant conversions plus
    # RAM churn at millions of cells). Convert once here; downstream then sees a dense ndarray.
    X_dense = norm_counts.X.toarray() if hasattr(norm_counts.X, "toarray") else np.asarray(norm_counts.X)
    by_k = OrderedDict()
    for idx in job_idx:
        p = run_params.iloc[idx, :]
        by_k.setdefault(int(p['n_components']), []).append((int(p['iter']), int(p['nmf_seed'])))

    for k, jobs in by_k.items():
        run_kwargs = dict(base_kwargs); run_kwargs['n_components'] = k
        for start in range(0, len(jobs), batch):
            chunk = jobs[start:start + batch]
            iters = [it for it, _ in chunk]
            seeds = [s for _, s in chunk]
            print('[Worker %d]. k=%d: launching %d replicate(s), iters=%s.'
                  % (worker_i, k, len(chunk), iters))
            results = _nmf_gpu_mu(X_dense, seeds, run_kwargs, gpu_kwargs)
            for (spectra, _usages), it in zip(results, iters):
                spectra = pd.DataFrame(spectra, index=np.arange(1, k + 1), columns=genes)
                save_df_to_npz(spectra, cnmf_obj.paths['iter_spectra'] % (k, it))
