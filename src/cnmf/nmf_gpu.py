#!/usr/bin/env python
"""GPU NMF kernel — raw-PyTorch Frobenius multiplicative-update (MU) factorization.

Standalone (no cnmf dependency); exposes a core NMF factorization and a thin `cNMF._nmf` adapter.

nmf_kwargs (from cnmf): reads n_components, max_iter, tol, random_state, init.
  Ignores beta_loss/solver/alpha_W/alpha_H/l1_ratio by design — always Frobenius MU, no regularization
  (matches cnmf defaults; would diverge if a run enables regularization). Defaults are centralized in
  `DEFAULT_NMF` and `DEFAULT_GPU`. Init runs through sklearn's _initialize_nmf for parity
  (random|nndsvd|nndsvda|nndsvdar; None uses `DEFAULT_NMF["init"]`; 'custom' unsupported).

gpu_kwargs (from Nextflow config): precedence gpu_kwargs dict > `DEFAULT_GPU`.
    device        auto|cuda|cuda:N|mps|cpu
    dtype         auto|fp32|fp64|bf16
    allow_tf32    TF32 for CUDA fp32 matmul
    compile       torch.compile the MU step      (CPU/CUDA; ignored on MPS)
    eps           MU denominator guard, stored in compute dtype
    check_every   iters/block when eager
    compile_block iters/block when compiled

Resolution — auto adapts silently; an explicit-but-unavailable value RAISES:
    device : auto = CUDA->MPS->CPU;            explicit cuda/mps unavailable -> error
    dtype  : auto = fp64 on CPU, fp32 on GPU;  explicit fp64 on MPS         -> error
             bf16 is explicit CUDA-only experimental storage + matmul dtype

    device   storage(default)
    ------   ----------------
    CPU      fp64    (stable CPU reference)
    CUDA     fp32    (bf16 is explicit experimental storage/matmul; TF32 is for fp32 matmul)
    MPS      fp32    (no fp64 in hardware)

Convergence (relative drop in ||X-WH|| < tol) is checked once per block, between blocks. Eager runs
short blocks (check_every) for a responsive dynamic stop; compiled runs fixed compile_block-sized
blocks (the host sync that reads the error stays outside the compiled region) and so converges
tighter for the same tol.

allow_tf32 is a CUDA fp32-matmul performance preference (not a storage dtype), scoped to the
factorization and restored on exit; it is silently ignored where unsupported. compile is allowed on
CPU/CUDA and ignored on MPS. TF32 under compile also flips float32_matmul_precision to 'high', the
knob the inductor backend honors. bf16 uses bf16 storage and bf16 matmul; it is not fp32 storage with
bf16 temporary casts. Input X must be non-negative & finite.
Output dtype contract: returned arrays are float64 for compatibility; numerical accuracy is set by
the compute dtype.
"""
import contextlib

import numpy as np


# ---------------------------------------------------------------------
# Defaults and option parsing
# ---------------------------------------------------------------------
_TRUTHY = {"1", "true", "yes", "on"}

DEFAULT_NMF = {
    "max_iter": 1000,
    "tol": 1e-4,
    "init": "random",
}

DEFAULT_GPU = {
    "device": "auto",
    "dtype": "auto",
    "allow_tf32": False,
    "compile": False,
    "eps": 1e-9,
    "check_every": 10,
    "compile_block": 1,
}


def parse_gpu_args(parser):
    """Register cNMF CLI flags for the optional PyTorch GPU NMF engine."""
    group = parser.add_argument_group("NMF engine options")
    group.add_argument("--engine", type=str.lower, choices=["cpu", "gpu"], help="[factorize] NMF engine to use (default cpu)")
    group.add_argument("--gpu-device", type=str, help="[factorize,gpu] Device for GPU NMF: auto, cpu, cuda, cuda:N, or mps")
    group.add_argument("--gpu-dtype", type=str.lower, choices=["auto", "fp32", "fp64", "bf16"], help="[factorize,gpu] Storage and matmul dtype for GPU NMF (default auto)")
    group.add_argument("--gpu-allow-tf32", action="store_const", const=True, help="[factorize,gpu] Allow TF32 for CUDA fp32 matrix multiplication")
    group.add_argument("--gpu-compile", action="store_const", const=True, help="[factorize,gpu] Enable torch.compile for the GPU NMF update step")
    group.add_argument("--gpu-eps", type=float, help="[factorize,gpu] Multiplicative-update denominator guard")
    group.add_argument("--gpu-check-every", type=int, help="[factorize,gpu] Eager-mode convergence check interval")
    group.add_argument("--gpu-compile-block", type=int, help="[factorize,gpu] Number of MU iterations per compiled block")
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
    ]
    if any(getattr(args, name) is not None for name in gpu_arg_names):
        commands = ", ".join(available_commands)
        raise ValueError(f"NMF engine/GPU options are only valid with: {commands}")


def configure_nmf_engine(cnmf_obj, engine="cpu", gpu_kwargs=None):
    """Configure a cNMF instance with an optional GPU NMF adapter."""
    if engine not in ("cpu", "gpu"):
        raise ValueError("engine must be 'cpu' or 'gpu'")
    if engine == "cpu":
        return cnmf_obj

    def _gpu_nmf(X, nmf_kwargs):
        nmf_kwargs = dict(nmf_kwargs)
        nmf_kwargs["engine"] = "gpu"
        nmf_kwargs["gpu"] = gpu_kwargs or {}
        return _nmf_gpu(cnmf_obj, X, nmf_kwargs)

    cnmf_obj._nmf = _gpu_nmf
    return cnmf_obj


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
    )


# ---------------------------------------------------------------------
# Runtime backend selection
# ---------------------------------------------------------------------
def _select_device(torch, requested):
    """auto = CUDA->MPS->CPU; an explicit GPU that's unavailable raises (never a silent slow-CPU)."""
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
    """auto = fp64 on CPU, fp32 on GPU; bf16 is explicit CUDA-only experimental storage."""
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
    """Scope CUDA TF32 fp32-matmul to `enable`, restoring the previous globals on exit (no-op off
    CUDA). Flips BOTH backends.cuda.matmul.allow_tf32 and float32_matmul_precision — the latter is
    what torch.compile's inductor backend honors. TF32 only affects fp32 matmul and is silently
    ignored by pre-Ampere hardware."""
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
# Input validation, loud imports, and initialization
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
    """sklearn-parity init via _initialize_nmf -> (W0, H0) as numpy. `init` None uses the centralized
    NMF default. Explicit random|nndsvd|nndsvda|nndsvdar pass through; 'custom' is unsupported
    (supply W/H yourself)."""
    if init == "custom":
        raise NotImplementedError("nmf_gpu does not support init='custom'")
    _initialize_nmf = _loud_import_initialize_nmf()      # private, but pinned by the cnmf-parity goal
    return _initialize_nmf(
        Xnp,
        n_components=k,
        init=(init or DEFAULT_NMF["init"]),
        random_state=seed,
    )


# ---------------------------------------------------------------------
# Multiplicative-update kernel and execution plan
# ---------------------------------------------------------------------
def _mu_step(W, H, Xg, eps):
    """One functional Frobenius MU update — W first (old H), then H (new W), matching the update
    order of sklearn's `_fit_multiplicative_update` (update W, then H using the just-updated W).
    Out-of-place (no `*=`) so torch.compile can trace it; grouped as factor*(num/den) for stable
    rounding. Σ_b over cell-blocks collapses to these dense matmuls."""
    W = W * ((Xg @ H.T) / (W @ (H @ H.T) + eps))           # W *= XHᵀ / (W·HHᵀ)   (uses old H)
    H = H * ((W.T @ Xg) / ((W.T @ W) @ H + eps))           # H *= WᵀX / (WᵀW·H)   (uses new W)
    return W, H


def _execution_plan(torch, opt, device):
    """Resolve compile policy and convergence-check cadence in one place.

    MPS does not support torch.compile reliably for this path, so compile requests are treated as
    eager there. CPU/CUDA compile is allowed. Compiled runs use compile_block because the host sync
    for convergence checks must stay outside the compiled step; eager runs use check_every for
    responsive stopping.
    """
    use_compile = opt["compile"] and not device.startswith("mps")
    if use_compile:
        return torch.compile(_mu_step), opt["compile_block"]
    return _mu_step, opt["check_every"]


def _to_device_factors(torch, W0, H0, dtype, device):
    """Move initialized numpy factors to the selected runtime backend."""
    W = torch.as_tensor(np.ascontiguousarray(W0), dtype=dtype, device=device)        # usages  (cells x k)
    H = torch.as_tensor(np.ascontiguousarray(H0), dtype=dtype, device=device)        # spectra (k x genes)
    return W, H


def _to_device_eps(torch, eps, dtype, device):
    """Move the MU denominator guard to the same dtype/device as X/W/H."""
    return torch.tensor(eps, dtype=dtype, device=device)


def _check_runtime_tensors(Xg, W, H, eps):
    """Ensure storage dtype is also the matmul dtype for the MU loop."""
    dtypes = {Xg.dtype, W.dtype, H.dtype, eps.dtype}
    if len(dtypes) != 1:
        raise RuntimeError(f"NMF runtime tensors must share dtype; got {sorted(map(str, dtypes))}")


def _fit_mu(torch, Xg, W, H, eps, max_iter, tol, step, block, tf32, device):
    """Run Frobenius MU updates until convergence or max_iter."""
    _check_runtime_tensors(Xg, W, H, eps)
    err_init = prev_err = None
    with torch.no_grad(), _cuda_tf32(torch, tf32, device):
        it = 0
        while it < max_iter:
            n = min(block, max_iter - it)
            for _ in range(n):                             # MU updates run inside the (compiled) step
                W, H = step(W, H, Xg, eps)
            it += n
            # stop on relative drop in direct ||X-WH|| < tol (host sync stays outside the compiled step)
            err = float(torch.linalg.norm(Xg - W @ H))
            if err_init is None:
                err_init = err
                if err_init == 0.0:                # degenerate (e.g. all-zero X): nothing to factor, avoid 0/0
                    break
            elif prev_err is not None and (prev_err - err) / err_init < tol:
                break
            prev_err = err
    return W, H


# ---------------------------------------------------------------------
# Public API and adapters
# ---------------------------------------------------------------------
def _to_nmf_output(H, W):
    """Return spectra/usages as numpy float64 for compatibility; compute precision is unchanged."""
    return H.cpu().double().numpy(), W.cpu().double().numpy()


def factorize_nmf_gpu(X, nmf_kwargs, gpu_kwargs=None):
    """Standalone Frobenius MU NMF: X -> (spectra, usages) = (H, W).

    CUDA is optional: CPU and MPS paths do not require CUDA. CUDA execution
    requires an installed PyTorch build where `torch.cuda.is_available()` is
    true; this function does not pin a CUDA toolkit/runtime version itself.
    Explicit `dtype='bf16'` additionally requires PyTorch to report
    `torch.cuda.is_bf16_supported()` for the selected CUDA device.
    """
    torch = _loud_import_torch()

    opt = _resolve_gpu_opts(gpu_kwargs)
    device = _select_device(torch, opt["device"])
    dtype = _select_storage(torch, opt["dtype"], device)

    k        = int(nmf_kwargs["n_components"])
    if k < 1:
        raise ValueError("n_components must be >= 1")
    max_iter = int(nmf_kwargs.get("max_iter", DEFAULT_NMF["max_iter"]))
    tol      = float(nmf_kwargs.get("tol", DEFAULT_NMF["tol"]))
    eps      = _to_device_eps(torch, opt["eps"], dtype, device)
    seed     = nmf_kwargs.get("random_state", None)

    Xnp = _to_checked_array(X)
    # TODO: sparse VRAM path. This prototype copies dense X to device; future row-blocked/sparse MU
    # should stream X blocks to GPU/MPS instead of requiring full dense X in VRAM.
    Xg = torch.as_tensor(Xnp, dtype=dtype, device=device)                            # cells x genes

    # sklearn-parity init on CPU (_initialize_nmf), then move to device (cnmf uses init='random')
    W0, H0 = _init_wh(Xnp, k, seed, nmf_kwargs.get("init"))
    W, H = _to_device_factors(torch, W0, H0, dtype, device)

    step, block = _execution_plan(torch, opt, device)
    tf32 = device.startswith("cuda") and opt["allow_tf32"] and dtype is torch.float32

    W, H = _fit_mu(torch, Xg, W, H, eps, max_iter, tol, step, block, tf32, device)
    return _to_nmf_output(H, W)


def _nmf_gpu(self, X, nmf_kwargs, gpu_kwargs=None):
    """cNMF adapter: same core NMF kernel, with `self` ignored for monkeypatch compatibility."""
    nmf_kwargs = dict(nmf_kwargs)
    embedded_gpu_kwargs = nmf_kwargs.pop("gpu", None)
    nmf_kwargs.pop("engine", None)
    if gpu_kwargs is None:
        gpu_kwargs = embedded_gpu_kwargs
    return factorize_nmf_gpu(X, nmf_kwargs, gpu_kwargs)
