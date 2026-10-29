"""FP64 CPU CD streaming refits and an opt-in cNMF adapter.

Inputs must already have the intended expression normalization, gene order and
scaling. Usages must have the same cell order as expression. No implicit TPM,
usage normalization, gene selection, or scaling is performed here.
"""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings

import anndata as ad
import h5py
import numpy as np
from scipy import sparse
from sklearn.decomposition import non_negative_factorization
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# Working-buffer budget, not a whole-process RAM limit. Leave 25% headroom.
MEMORY_BUDGET = 16 * 1024**3
DEFAULT_THREADS = 4
DEFAULT_WORKERS = 4


def gene_stats_streaming(expression):
    """Population mean/variance, matching cNMF get_mean_var without loading X."""
    scaler = StandardScaler(with_mean=False)
    rows = _chunk_size(getattr(expression, 'source_shape', expression.shape), 0)
    for _, _, block in _blocks(expression, rows):
        scaler.partial_fit(block)
    return scaler.mean_, scaler.var_


@threadpool_limits.wrap(limits=DEFAULT_THREADS, user_api='blas')
def ols_streaming(usages, expression):
    """cNMF's normalized OLS, with streamed gene statistics and cross-products."""
    mean, variance = gene_stats_streaming(expression)
    std = np.sqrt(np.maximum(variance, 1e-12))
    k = usages.shape[1]
    rows = _chunk_size(expression.shape, k, usages.nbytes + 8*k*expression.shape[1])
    xtx = np.zeros((k, k), dtype=np.float64)
    xty = np.zeros((k, expression.shape[1]), dtype=np.float64)
    for start, stop, block in _blocks(expression, rows):
        if sparse.issparse(block):
            block = block.toarray()
        block = (block - mean) / std
        w = np.asarray(usages[start:stop], dtype=np.float64)
        xtx += w.T @ w
        xty += w.T @ block
    return np.linalg.lstsq(xtx, xty, rcond=None)[0]


class ScaledExpression:
    """Lazy HVG selection and sample-standard-deviation scaling for final refit."""
    def __init__(self, expression, columns):
        if getattr(expression, 'format', None) == 'csc' and not sparse.issparse(expression):
            raise ValueError('Streaming currently requires CSR or dense X, not CSC')
        self.expression = expression
        self.columns = np.asarray(columns)
        self.shape = (expression.shape[0], len(columns))
        self.source_shape = expression.shape
        self.dtype = np.dtype('float64')
        self.scale = np.ones(len(columns))
        # Account for full-width sparse row reads when budgeting chunks.
        _, variance = gene_stats_streaming(self)
        self.scale = np.sqrt(variance * self.shape[0] / (self.shape[0] - 1))
        # Scanpy's sparse scale leaves constant genes unchanged.
        if sparse.issparse(expression) or getattr(expression, 'format', None) in ('csr', 'csc'):
            self.scale[self.scale == 0] = 1

    def __getitem__(self, rows):
        block = self.expression[rows][:, self.columns].astype(np.float64, copy=False)
        if sparse.issparse(block):
            return block.multiply(1 / self.scale).tocsr()
        return block / self.scale


def _chunk_size(shape, k, fixed_bytes=0):
    """Budget FP64 blocks, residuals, solver copies and fixed factors."""
    available = int(MEMORY_BUDGET * .75) - fixed_bytes
    bytes_per_row = DEFAULT_WORKERS * 8 * (6 * shape[1] + 4 * k)
    if available < bytes_per_row:
        raise ValueError('Streaming working-memory budget cannot fit one row')
    return min(max(1, (shape[0] + DEFAULT_WORKERS - 1) // DEFAULT_WORKERS),
               available // bytes_per_row)


@contextmanager
def _expression(path):
    """Open only X, not AnnData metadata, layers or embeddings."""
    if not isinstance(path, (str, Path)):
        yield path
        return
    path = Path(path)
    if path.suffix == '.npy':
        x = np.load(path, mmap_mode='r')
        yield x
    else:
        with h5py.File(path, 'r') as handle:
            x = handle['X']
            if isinstance(x, h5py.Group):
                x = ad.io.sparse_dataset(x)
                if x.format != 'csr':
                    raise ValueError('Streaming currently requires CSR or dense X, not CSC')
            yield x


def _checked(value):
    values = value.data if sparse.issparse(value) else value
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Refit inputs must be finite and nonnegative')
    return value


def _blocks(x, chunk_size, dtype=np.float64):
    """Read checked row chunks; dtype=None preserves storage precision for export."""
    if getattr(x, 'format', None) == 'csc' and not sparse.issparse(x):
        raise ValueError('Streaming currently requires CSR or dense X, not CSC')
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        block = x[start:stop]
        if dtype is not None:
            block = block.astype(dtype, copy=False)
        yield start, stop, _checked(block)


def _validate(chunk_size, max_iter, tol, alpha, l1_ratio):
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer')
    if max_iter < 1 or tol < 0 or alpha < 0 or not 0 <= l1_ratio <= 1:
        raise ValueError('Invalid iteration, tolerance or regularization settings')


def _map_blocks(function, x, chunk_size):
    """Read on the caller thread; bound loaded chunks and preserve reduction order."""
    blocks = _blocks(x, chunk_size)
    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as pool:
        while True:
            batch = list(islice(blocks, DEFAULT_WORKERS))
            if not batch:
                break
            futures = [pool.submit(function, *block) for block in batch]
            for future in futures:
                yield future.result()
            # Release inputs/results before reading the next batch (no eager prefetch).
            del batch, futures, future


def _refit_from_cnmf(obj, expression, fixed_factor, spectra=False):
    """Adapter for existing cNMF refits; no preprocessing or engine mutation."""
    import yaml
    with open(obj.paths['nmf_run_parameters']) as stream:
        options = yaml.safe_load(stream)
    if options.get('solver', 'mu') != 'cd' or options.get('beta_loss', 'frobenius') != 'frobenius':
        raise ValueError('Streaming refits require CD with Frobenius loss')
    if options.get('shuffle', False):
        raise ValueError('Streaming refits currently require shuffle=False')
    fixed_factor = np.asarray(fixed_factor)
    function = refit_spectra_streaming if spectra else refit_usage_streaming
    with TemporaryDirectory(prefix='cnmf-streaming-', dir=Path(obj.paths['nmf_run_parameters']).parent) as folder:
        output = Path(folder) / 'factor.npy'
        function(expression, fixed_factor, output,
                 max_iter=options.get('max_iter', 1000), tol=options.get('tol', 1e-4),
                 alpha=options.get('alpha_W', 0.0), l1_ratio=options.get('l1_ratio', 0.0))
        return np.load(output)


@threadpool_limits.wrap(limits=DEFAULT_THREADS, user_api='blas')
def refit_usage_streaming(expression_path, spectra, output_path, chunk_size=None,
                          max_iter=1000, tol=1e-4, alpha=0.0, l1_ratio=0.0):
    """
    Fit full-data usages with fixed spectra using CPU CD and save an NPY file.

    spectra : array
        Fixed K-by-genes matrix, in the input expression's gene order.
    alpha : float
        sklearn alpha_W: penalty scales with the number of expression genes.

    Returns a receipt with chunk iteration counts. Output is published only
    after every chunk succeeds; existing output is never overwritten.
    """
    h = _checked(np.array(spectra, dtype=np.float64, order='C', copy=True))
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    with _expression(expression_path) as x:
        if len(x.shape) != 2 or min(x.shape) < 1 or h.ndim != 2 or h.shape[0] < 1 or h.shape[1] != x.shape[1]:
            raise ValueError('Expression and spectra shapes do not match')
        if chunk_size is None:
            chunk_size = _chunk_size(getattr(x, 'source_shape', x.shape), h.shape[0],
                                     DEFAULT_WORKERS * (4*h.nbytes + 8*h.shape[0]**2))
        _validate(chunk_size, max_iter, tol, alpha, l1_ratio)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iterations = []
        with TemporaryDirectory(prefix='.refit-', dir=output_path.parent) as folder:
            temporary = Path(folder) / 'usages.npy'
            w = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float64,
                                         shape=(x.shape[0], h.shape[0]))
            def fit_block(start, stop, block):
                fitted, _, n_iter = non_negative_factorization(
                    block, H=h, update_H=False, n_components=h.shape[0],
                    solver='cd', beta_loss='frobenius', max_iter=max_iter, tol=tol,
                    alpha_W=alpha, alpha_H=0.0, l1_ratio=l1_ratio,
                    shuffle=False, random_state=0)
                w[start:stop] = fitted
                return int(n_iter)
            iterations.extend(_map_blocks(fit_block, x, chunk_size))
            w.flush()
            del w
            temporary.replace(output_path)
    return dict(output=str(output_path), iterations=iterations, dtype='float64')


@threadpool_limits.wrap(limits=DEFAULT_THREADS, user_api='blas')
def refit_spectra_streaming(expression_path, usages, output_path, chunk_size=None,
                            max_iter=1000, tol=1e-4, alpha=0.0, l1_ratio=0.0):
    """
    Fit full-gene spectra with W fixed, keeping W in RAM and streaming X.

    usages : array
        Cells-by-K matrix in expression row order. Apply cNMF's usage
        normalization before calling when reproducing its TPM-spectrum refit.
    alpha : float
        Equivalent to sklearn alpha_W on the transposed fixed-H problem
        refit_usage(X.T, usages.T); penalty scales with the full cell count.

    Uses sklearn's private CD sweep kernel; parity tests guard compatibility.
    Saves K-by-genes FP64 NPY, returning convergence and direct streamed SSE.
    """
    from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast

    w = _checked(np.asarray(usages, dtype=np.float64, order='C'))
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    with _expression(expression_path) as x:
        if len(x.shape) != 2 or min(x.shape) < 1 or w.ndim != 2 or w.shape[1] < 1 or w.shape[0] != x.shape[0]:
            raise ValueError('Expression and usages shapes do not match')
        if chunk_size is None:
            chunk_size = _chunk_size(x.shape, w.shape[1],
                                     w.nbytes + (32 + 8*DEFAULT_WORKERS)*x.shape[1]*w.shape[1]
                                     + 8*w.shape[1]**2)
        _validate(chunk_size, max_iter, tol, alpha, l1_ratio)
        a = np.ascontiguousarray(w.T @ w)
        # Transposed layout matches sklearn's fixed-H refit on X.T.
        b = np.zeros((x.shape[1], w.shape[1]), dtype=np.float64)
        def cross_product(start, stop, block):
            return block.T @ w[start:stop]
        for partial in _map_blocks(cross_product, x, chunk_size):
            b += partial
        a.flat[::w.shape[1]+1] += x.shape[0] * alpha * (1-l1_ratio)
        b -= x.shape[0] * alpha * l1_ratio
        ht = np.zeros_like(b)
        order = np.arange(w.shape[1], dtype=np.intp)
        converged = False
        for n_iter in range(1, max_iter+1):
            violation = _update_cdnmf_fast(ht, a, b, order)
            if n_iter == 1:
                initial_violation = violation
            if initial_violation == 0 or violation / initial_violation <= tol:
                converged = True
                break
        if not converged and tol > 0:
            warnings.warn('Streaming spectra refit reached max_iter', ConvergenceWarning)
        h = ht.T
        # Direct residual evaluation avoids cancellation in the statistics formula.
        def squared_error(start, stop, block):
            residual = w[start:stop] @ h
            if sparse.issparse(block):
                block = block.toarray()
            residual -= block
            return float(np.square(residual).sum(dtype=np.float64))
        sse = sum(_map_blocks(squared_error, x, chunk_size))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.refit-', dir=output_path.parent) as folder:
            temporary = Path(folder) / 'spectra.npy'
            np.save(temporary, h)
            temporary.replace(output_path)
    return dict(output=str(output_path), iterations=n_iter, converged=converged,
                SSE=sse, dtype='float64')


@threadpool_limits.wrap(limits=DEFAULT_THREADS, user_api='blas')
def reconstruction_stats_streaming(expression, usages, spectra):
    """Original global-mean R²/SSE/TSS, with two FP64 row-streamed passes."""
    h = np.asarray(spectra, dtype=np.float64)
    with _expression(expression) as x:
        rows = _chunk_size(x.shape, h.shape[0], h.nbytes + usages.nbytes)
        def block_sum(start, stop, block):
            return float(block.sum(dtype=np.float64))
        total = sum(_map_blocks(block_sum, x, rows))
        mean = total / (x.shape[0] * x.shape[1])
        def block_errors(start, stop, block):
            if sparse.issparse(block):
                block = block.toarray()
            residual = np.asarray(usages[start:stop], dtype=np.float64) @ h
            residual -= block
            np.square(residual, out=residual)
            block_sse = float(residual.sum(dtype=np.float64))
            centered = block - mean
            np.square(centered, out=centered)
            return block_sse, float(centered.sum(dtype=np.float64))
        sse = tss = 0.0
        for block_sse, block_tss in _map_blocks(block_errors, x, rows):
            sse += block_sse
            tss += block_tss
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1 - np.divide(sse, tss)
    return r2, sse, tss
