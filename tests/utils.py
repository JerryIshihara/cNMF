"""Reusable helpers and fixtures for Python tests."""

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest


# torch + sklearn/scipy can double-load OpenMP on macOS; tolerate it for this test process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = REPO_ROOT / "src" / "cnmf" / "nmf_gpu.py"
DOWNLOAD_PYTEST_DATA_PATH = REPO_ROOT / "download_pytest_data.py"


# ---------------------------------------------------------------------
# Standalone NMF GPU kernel helpers
# ---------------------------------------------------------------------
def load_kernel_module(module_name="nmf_gpu", kernel_path=KERNEL_PATH):
    """Load the standalone kernel script as an importable module for tests."""
    kernel_path = Path(kernel_path)
    if not kernel_path.exists():
        pytest.fail(f"Required NMF GPU kernel file is missing: {kernel_path}", pytrace=False)

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Unable to load {module_name!r} from {kernel_path}", pytrace=False)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def kernel():
    """Loaded `src/cnmf/nmf_gpu.py` module under test."""
    return load_kernel_module()


def require_nmf_runtime():
    """Skip tests that need the lazy torch + sklearn runtime dependencies."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    return torch


def low_rank_matrix(seed=0, cells=40, genes=20, rank=3):
    """Build an exact non-negative rank-k matrix for reconstruction tests."""
    rng = np.random.default_rng(seed)
    W = rng.random((cells, rank)) + 0.2
    H = rng.random((rank, genes)) + 0.2
    return W @ H


def small_nonnegative_matrix(seed=1, cells=10, genes=8):
    """Build a small dense non-negative matrix for quick smoke tests."""
    rng = np.random.default_rng(seed)
    return rng.random((cells, genes)) + 0.1


def assert_valid_nmf_output(X, H, W, k):
    """Assert the shared cNMF-compatible output contract for NMF factors."""
    assert H.shape == (k, X.shape[1])
    assert W.shape == (X.shape[0], k)
    assert H.dtype == np.float64
    assert W.dtype == np.float64
    assert np.isfinite(H).all()
    assert np.isfinite(W).all()
    assert (H >= 0).all()
    assert (W >= 0).all()


def fake_torch_backend(cuda_available=False, mps_available=False, bf16_supported=True):
    """Create a tiny fake torch surface for backend policy tests."""
    return SimpleNamespace(
        float32="float32",
        float64="float64",
        bfloat16="bfloat16",
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            is_bf16_supported=lambda: bf16_supported,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available),
        ),
    )


# ---------------------------------------------------------------------
# Reproducibility fixture configuration
# ---------------------------------------------------------------------
REPRODUCIBILITY_FIXTURE_CONFIGS = {
    "simulated": {
        "data_dir": Path("simulated_example_data"),
        "name": "example_cNMF",
        "counts_file": Path("filtered_counts.txt"),
        "components": np.arange(5, 8),
        "n_iter": 15,
        "num_highvar_genes": 1000,
        "seed": 14,
        "consensus": [(7, 0.1)],
    },
    "pbmc": {
        "data_dir": Path("example_PBMC"),
        "name": "pbmc_cNMF",
        "counts_file": Path("counts.h5ad"),
        "components": np.arange(7, 10),
        "n_iter": 15,
        "num_highvar_genes": 1000,
        "seed": 14,
        "consensus": [(7, 0.1), (8, 0.1)],
    },
}
REPRODUCIBILITY_FIXTURE_RUN_DIRS = tuple(
    cfg["data_dir"] / cfg["name"] for cfg in REPRODUCIBILITY_FIXTURE_CONFIGS.values()
)
DEFAULT_TEST_CACHE_DIR = Path(os.environ.get("CNMF_TEST_CACHE_DIR", "tests/.cache"))
DEFAULT_TEST_DATA_DIR = Path(os.environ.get("CNMF_TEST_DATA_DIR", "tests/test_data"))
DEFAULT_PYTEST_CACHE_DIR = Path(".pytest_cache")


def default_test_log_path():
    """Return a per-run pytest log path named with UTC timestamp plus UUID."""
    explicit_path = os.environ.get("CNMF_TEST_LOG")
    if explicit_path:
        return Path(explicit_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_TEST_CACHE_DIR / f"pytest-{timestamp}-{uuid.uuid4().hex}.log"


def reproducibility_fixture_config(dataset):
    """Return a copy of the named reproducibility fixture config."""
    try:
        return dict(REPRODUCIBILITY_FIXTURE_CONFIGS[dataset])
    except KeyError as exc:
        choices = "|".join(sorted(REPRODUCIBILITY_FIXTURE_CONFIGS))
        raise ValueError(f"Unknown reproducibility fixture {dataset!r}; use {choices}.") from exc


# ---------------------------------------------------------------------
# Reproducibility fixture generation
# ---------------------------------------------------------------------
def regenerate_reproducibility_fixture(dataset, data_root=Path("tests/test_data"), overwrite=False,
                                       total_workers=1):
    """Regenerate one cNMF reproducibility fixture under tests/test_data.

    This is intentionally not used by pytest collection. Call it manually when
    refreshing the downloaded reference bundle for a dependency stack.
    Input counts must already exist under `data_root`; this helper only reruns
    cNMF outputs and consensus references.
    """
    cfg = reproducibility_fixture_config(dataset)
    data_root = Path(data_root)
    output_dir = data_root / cfg["data_dir"]
    counts_fn = output_dir / cfg["counts_file"]
    run_dir = output_dir / cfg["name"]

    if not counts_fn.exists():
        raise FileNotFoundError(f"Missing counts input for {dataset!r}: {counts_fn}")
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} already exists; pass overwrite=True to regenerate it.")
        import shutil
        shutil.rmtree(run_dir)

    from cnmf import cNMF

    cnmf_obj = cNMF(output_dir=str(output_dir), name=cfg["name"])
    cnmf_obj.prepare(
        counts_fn=str(counts_fn),
        components=cfg["components"],
        n_iter=cfg["n_iter"],
        num_highvar_genes=cfg["num_highvar_genes"],
        seed=cfg["seed"],
        # Preserve the solver used to create the historical reference bundle.
        solver="cd",
    )
    if total_workers == 1:
        cnmf_obj.factorize(worker_i=0, total_workers=1)
    else:
        cnmf_obj.factorize_multi_process(total_workers)
    cnmf_obj.combine()
    for k, density_threshold in cfg["consensus"]:
        cnmf_obj.consensus(
            k=k,
            density_threshold=density_threshold,
            show_clustering=False,
            close_clustergram_fig=False,
        )
    return cnmf_obj


# ---------------------------------------------------------------------
# Reproducibility fixture session state
# ---------------------------------------------------------------------
def reproducibility_fixtures_present(data_root=Path("tests/test_data")):
    """Return whether the downloaded reproducibility fixture inputs are present."""
    data_root = Path(data_root)
    return all((data_root / cfg["data_dir"]).exists()
               for cfg in REPRODUCIBILITY_FIXTURE_CONFIGS.values())


def download_reproducibility_fixture_data(data_root=Path("tests/test_data")):
    """Download the upstream pytest fixture bundle into the default test-data location."""
    data_root = Path(data_root)
    default_data_root = Path("tests/test_data")
    if data_root != default_data_root:
        raise ValueError(
            "download_pytest_data.py writes to tests/test_data; custom CNMF_TEST_DATA_DIR "
            "is only supported when local fixture archives already exist."
        )

    spec = importlib.util.spec_from_file_location("download_pytest_data", DOWNLOAD_PYTEST_DATA_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load fixture downloader: {DOWNLOAD_PYTEST_DATA_PATH}")

    print("Fixture data not found; downloading pytest fixture data.")
    download_pytest_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(download_pytest_data)
    download_pytest_data.main()


def backup_reproducibility_outputs(data_root, backup_dir):
    """Move existing reproducibility output directories aside before regeneration."""
    data_root = Path(data_root)
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for fixture_run_dir in REPRODUCIBILITY_FIXTURE_RUN_DIRS:
        src = data_root / fixture_run_dir
        dst = backup_dir / fixture_run_dir
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved.append(str(fixture_run_dir))
    return moved


def restore_reproducibility_outputs(data_root, backup_dir):
    """Restore fixture outputs that were moved aside before regeneration."""
    data_root = Path(data_root)
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return

    for fixture_run_dir in REPRODUCIBILITY_FIXTURE_RUN_DIRS:
        generated = data_root / fixture_run_dir
        backed_up = backup_dir / fixture_run_dir
        if generated.exists():
            shutil.rmtree(generated)
        if backed_up.exists():
            generated.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backed_up), str(generated))
    shutil.rmtree(backup_dir, ignore_errors=True)


def _write_fixture_manifest(manifest_path, manifest):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def ensure_reproducibility_fixture_state(data_root=Path("tests/test_data"),
                                         manifest_path=Path("tests/.cache/reproducibility-manifest.json"),
                                         backup_dir=None):
    """Ensure reproducibility fixtures match the active dependency stack.

    This creates a manifest so pytest cleanup can delete downloaded fixture data
    or restore pre-existing fixture outputs after the test session exits.
    """
    data_root = Path(data_root)
    manifest_path = Path(manifest_path)
    backup_dir = Path(backup_dir) if backup_dir is not None else (
        manifest_path.parent / f"reproducibility-backup-{os.getpid()}"
    )

    manifest = {
        "backup_dir": "",
        "data_root": str(data_root),
        "generated_test_data": False,
    }
    _write_fixture_manifest(manifest_path, manifest)

    if not reproducibility_fixtures_present(data_root):
        manifest["generated_test_data"] = True
        _write_fixture_manifest(manifest_path, manifest)
        download_reproducibility_fixture_data(data_root=data_root)

    if not manifest["generated_test_data"]:
        backup_reproducibility_outputs(data_root=data_root, backup_dir=backup_dir)
        manifest["backup_dir"] = str(backup_dir)
        _write_fixture_manifest(manifest_path, manifest)

    for dataset in REPRODUCIBILITY_FIXTURE_CONFIGS:
        print(f"Regenerating fixture data: {dataset}")
        regenerate_reproducibility_fixture(dataset, data_root=data_root, overwrite=True)

    return manifest


def cleanup_reproducibility_fixture_state(manifest_path, keep_test_data=False):
    """Clean generated reproducibility artifacts recorded in a runner manifest."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text())
    data_root = Path(manifest["data_root"])

    if manifest.get("generated_test_data") and not keep_test_data and data_root.exists():
        print(f"Deleting generated test data: {data_root}")
        shutil.rmtree(data_root)

    backup_dir = manifest.get("backup_dir")
    if backup_dir:
        restore_reproducibility_outputs(data_root=data_root, backup_dir=Path(backup_dir))

    manifest_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Pytest collection/session helpers
# ---------------------------------------------------------------------
def configure_test_cache_dirs(cache_dir=DEFAULT_TEST_CACHE_DIR):
    """Route common test caches into the ignored local test cache directory."""
    cache_dir = Path(cache_dir)
    for subdir in ("matplotlib", "numba", "xdg"):
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


def pytest_items_need_reproducibility_fixtures(items):
    """Return whether collected pytest items include the reproducibility suite."""
    for item in items:
        item_path = Path(str(getattr(item, "path", getattr(item, "fspath", ""))))
        if item_path.name == "test_reproducibility.py":
            return True
    return False


def prepare_reproducibility_fixtures_for_pytest(config, items):
    """Prepare active-stack reproducibility fixtures if pytest collected that suite."""
    if not pytest_items_need_reproducibility_fixtures(items):
        return

    manifest_path = DEFAULT_TEST_CACHE_DIR / f"reproducibility-manifest-{os.getpid()}.json"
    config._cnmf_reproducibility_manifest = manifest_path
    ensure_reproducibility_fixture_state(
        data_root=DEFAULT_TEST_DATA_DIR,
        manifest_path=manifest_path,
    )


def cleanup_reproducibility_fixtures_for_pytest(config):
    """Clean temporary reproducibility artifacts created for this pytest session."""
    manifest_path = getattr(config, "_cnmf_reproducibility_manifest", None)
    if manifest_path is None:
        return

    print()
    cleanup_reproducibility_fixture_state(
        manifest_path=manifest_path,
        keep_test_data=os.environ.get("CNMF_KEEP_TEST_DATA", "0") == "1",
    )


# ---------------------------------------------------------------------
# Pytest log and cache cleanup
# ---------------------------------------------------------------------
def write_pytest_log(terminalreporter, exitstatus, log_path=None):
    """Write a compact pytest result log under tests/.cache."""
    log_path = Path(log_path) if log_path is not None else default_test_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats = terminalreporter.stats
    outcomes = ("failed", "error", "passed", "skipped", "xfailed", "xpassed")
    lines = [
        "cNMF pytest log",
        f"log_file: {log_path}",
        f"exitstatus: {exitstatus}",
        f"collected: {getattr(terminalreporter, '_numcollected', 'unknown')}",
        "",
        "summary:",
    ]

    details = []
    for outcome in outcomes:
        reports = stats.get(outcome, [])
        selected = [
            report for report in reports
            if outcome == "skipped" or getattr(report, "when", "call") == "call"
        ]
        lines.append(f"  {outcome}: {len(selected)}")
        for report in selected:
            duration = getattr(report, "duration", None)
            if duration is None:
                details.append(f"{outcome.upper()} {report.nodeid}")
            else:
                details.append(f"{outcome.upper()} {report.nodeid} ({duration:.3f}s)")

    lines.extend(["", "details:", *details, ""])
    log_path.write_text("\n".join(lines))


def cleanup_runtime_cache_dirs(cache_dir=DEFAULT_TEST_CACHE_DIR,
                               pytest_cache_dir=DEFAULT_PYTEST_CACHE_DIR):
    """Remove runtime cache subdirs while leaving timestamped pytest logs."""
    if os.environ.get("CNMF_KEEP_TEST_CACHE", "0") == "1":
        return

    cache_dir = Path(cache_dir)
    for path in (
        cache_dir / "matplotlib",
        cache_dir / "numba",
        cache_dir / "reproducibility",
        cache_dir / "xdg",
        Path(pytest_cache_dir),
    ):
        if path.exists():
            print(f"Deleting test cache: {path}")
            shutil.rmtree(path, ignore_errors=True)

    for path in cache_dir.glob("reproducibility-manifest-*.json"):
        cleanup_reproducibility_fixture_state(path)
    for path in cache_dir.glob("reproducibility-backup-*"):
        print(f"Deleting test cache: {path}")
        shutil.rmtree(path, ignore_errors=True)

    legacy_log = cache_dir / "pytest.log"
    if "CNMF_TEST_LOG" not in os.environ and legacy_log.exists():
        print(f"Deleting test cache: {legacy_log}")
        legacy_log.unlink(missing_ok=True)
