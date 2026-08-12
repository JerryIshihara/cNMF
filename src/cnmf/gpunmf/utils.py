"""Shared configuration and runtime helpers for the GPU NMF solvers."""

import contextlib
from collections import namedtuple

import numpy as np


_TRUTHY = {"1", "true", "yes", "on"}

DEFAULT_NMF = {
    "max_iter": 1000,
    "tol": 1e-4,
    "init": "random",
    "solver": "cd",
}


DEFAULT_GPU = {
    "device": "auto",
    "dtype": "auto",
    "allow_tf32": False,
    "compile": False,
    "eps": float(np.finfo(np.float32).eps),
    "check_every": 10,
    "compile_block": 1,
    "batch": 1,
    "row_tiling_ratio": None,
}


GPU_ARG_NAMES = (
    "engine",
    "gpu_device",
    "gpu_dtype",
    "gpu_allow_tf32",
    "gpu_compile",
    "gpu_eps",
    "gpu_check_every",
    "gpu_compile_block",
    "gpu_batch",
    "gpu_row_tiling_ratio",
)


# ---------------------------------------------------------------------
# Engine argument validation and solver contract enforcement
# ---------------------------------------------------------------------

def _validate_engine_args(args, available_solvers):
    # sequencial validation of engine args for the given command
    _validate_engine_args_for_command(args)
    _validate_nmf_solver(args.solver, args.beta_loss, available_solvers)


def _validate_engine_args_for_command(args, available_commands=("prepare", "factorize", "consensus")):
    """Engine/GPU CLI options are only valid for commands that support the selected engine."""
    available_commands = tuple(available_commands)
    if args.command in available_commands:
        return

    if any(getattr(args, name) is not None for name in GPU_ARG_NAMES):
        commands = ", ".join(available_commands)
        raise ValueError(f"NMF engine/GPU options are only valid with: {commands}")


def _validate_nmf_solver(solver, beta_loss, available_solvers):
    """Normalize a solver and enforce its loss-function contract."""
    available_solvers = tuple(available_solvers)
    solver = str(solver).strip().lower()
    if solver not in available_solvers:
        available = ", ".join(sorted(available_solvers))
        raise ValueError(
            f"solver must be one of: {available}"
        )
    if solver == "cd" and not (
        beta_loss == 2 or str(beta_loss).strip().lower() == "frobenius"
    ):
        raise ValueError("solver='cd' supports only beta_loss='frobenius'")
    return solver

# ---------------------------------------------------------------------
# CLI argument parsing and option resolution
# ---------------------------------------------------------------------

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
        "row_tiling_ratio": args.gpu_row_tiling_ratio,
    }
    if args.engine != "gpu":
        if any(value is not None for value in raw.values()):
            raise ValueError("GPU options require --engine gpu")
        return None
    return _resolve_gpu_opts(raw)



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

    def parse_optional_tiling_ratio(value):
        return None if value is None else _validate_row_tiling_ratio(value)

    return dict(
        device        = parse_typed(raw.get("device"), DEFAULT_GPU["device"], str, str.lower),
        dtype         = parse_typed(raw.get("dtype"), DEFAULT_GPU["dtype"], str, str.lower),
        allow_tf32    = parse_bool(raw.get("allow_tf32"), DEFAULT_GPU["allow_tf32"]),
        compile       = parse_bool(raw.get("compile"), DEFAULT_GPU["compile"]),
        eps           = parse_typed(raw.get("eps"), DEFAULT_GPU["eps"], float),
        check_every   = parse_positive_int(raw.get("check_every"), DEFAULT_GPU["check_every"]),
        compile_block = parse_positive_int(raw.get("compile_block"), DEFAULT_GPU["compile_block"]),
        batch         = parse_positive_int(raw.get("batch"), DEFAULT_GPU["batch"]),
        row_tiling_ratio = parse_optional_tiling_ratio(raw.get("row_tiling_ratio")),
    )


# ---------------------------------------------------------------------
# GPU row tiling
# ---------------------------------------------------------------------


_MIB = 1 << 20


def _validate_row_tiling_ratio(value):
    """Return a finite configured row ratio in the interval (0, 1]."""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            "gpu row tiling ratio must be a finite number in (0, 1]"
        ) from None
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError(
            "gpu row tiling ratio must be a finite number in (0, 1]"
        )
    return ratio


def _check_row_batch(row_batch, n_rows):
    """Validate and cap a resolved row batch."""
    if row_batch < 1 or n_rows < 1:
        raise ValueError("gpu row batch and row count must be at least 1")
    return min(n_rows, row_batch)


def resolve_row_batch(
    torch,
    device,
    Xcompute,
    replicates,
    n_components,
    *,
    configured_ratio=None,
    reserve_bytes=512 * _MIB,
    reserve_fraction=0.10,
):
    """Resolve a configured row ratio or automatically size one from VRAM."""
    n_rows, _n_features = Xcompute.shape
    if n_rows < 1:
        raise ValueError("fixed-H input must contain at least one row")

    if configured_ratio is not None:
        tiling_ratio = _validate_row_tiling_ratio(configured_ratio)
        row_batch = max(1, int(n_rows * tiling_ratio))
        return _check_row_batch(row_batch, n_rows), tiling_ratio

    tiling_ratio = 1.0
    if str(device).split(":", 1)[0] == "cuda":
        if reserve_bytes < 0 or not 0 <= reserve_fraction < 1:
            raise ValueError("invalid VRAM reserve")
        free_vram_bytes, _ = torch.cuda.mem_get_info(device)
        factor_bytes = (
            replicates
            * n_components
            * n_rows
            * Xcompute.dtype.itemsize
        )
        required_bytes = Xcompute.nbytes + 2 * factor_bytes
        reserve = max(
            reserve_bytes,
            int(free_vram_bytes * reserve_fraction),
        )
        usable_vram_bytes = max(0, free_vram_bytes - reserve)
        if required_bytes > usable_vram_bytes:
            bytes_per_row = (required_bytes + n_rows - 1) // n_rows
            if usable_vram_bytes < bytes_per_row:
                raise MemoryError(
                    "GPU fixed-H CD does not have enough VRAM for one row"
                )
            portions = (
                required_bytes + usable_vram_bytes - 1
            ) // usable_vram_bytes
            tiling_ratio = 1 / portions

    row_batch = max(1, int(n_rows * tiling_ratio))
    return _check_row_batch(row_batch, n_rows), tiling_ratio


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
        raise NotImplementedError("GPU NMF does not support init='custom'")
    _initialize_nmf = _loud_import_initialize_nmf()
    return _initialize_nmf(
        Xnp,
        n_components=k,
        init=(init or DEFAULT_NMF["init"]),
        random_state=seed,
    )


# ---------------------------------------------------------------------
# Shared solver runtime and device-tensor helpers
# ---------------------------------------------------------------------


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


def _to_device_eps(torch, eps, dtype, device):
    """Create the exact-zero denominator replacement on runtime dtype/device."""
    return torch.tensor(eps, dtype=dtype, device=device)


def _check_runtime_tensors(Xg, W, H, eps):
    """Require one dtype across X/W/H/eps."""
    dtypes = {Xg.dtype, W.dtype, H.dtype, eps.dtype}
    if len(dtypes) != 1:
        raise RuntimeError(f"NMF runtime tensors must share dtype; got {sorted(map(str, dtypes))}")


def _normalize_seeds(seeds):
    """Return a non-empty list of integer or None replicate seeds."""
    normalized = [None if seed is None else int(seed) for seed in seeds]
    if not normalized:
        raise ValueError("seeds must be a non-empty list of per-replicate random states")
    return normalized


def _recon_err(Xg, W, H, xnorm2):
    """Per-replicate ‖X − WH‖_F without materializing the full residual."""
    Wt = W.transpose(-2, -1)
    cross = ((Wt @ Xg) * H).sum(dim=(-2, -1))
    whnorm = ((Wt @ W) * (H @ H.transpose(-2, -1))).sum(dim=(-2, -1))
    return (xnorm2 - 2.0 * cross + whnorm).clamp_min(0).sqrt()


def _sq_norm(X):
    """Return ‖X‖² using row chunks to avoid BLAS vector-length limits."""
    rows = X.shape[0]
    step = max(1, (1 << 28) // max(1, X.numel() // max(1, rows)))
    total = X.new_zeros(())
    for i in range(0, rows, step):
        total = total + X[i:i + step].square().sum()
    return total


def _execution_plan(torch, opt, device, step_fn):
    """Return the solver step and convergence-check block for eager or compiled execution."""
    use_compile = opt["compile"] and not device.startswith("mps")
    if use_compile:
        return torch.compile(step_fn), opt["compile_block"]
    return step_fn, opt["check_every"]
