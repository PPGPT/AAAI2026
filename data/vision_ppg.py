import os
import numpy as np
from scipy.signal import spectrogram, resample

def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = (float(a.min()), float(a.max()))
    if hi - lo < 1e-08:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)

def _gaf(x: np.ndarray, size: int):
    xs = resample(x, size).astype(np.float64)
    xs = _minmax(xs) * 2.0 - 1.0
    xs = np.clip(xs, -1.0, 1.0)
    phi = np.arccos(xs)
    cos_i = np.cos(phi)
    sin_i = np.sin(phi)
    gasf = np.outer(cos_i, cos_i) - np.outer(sin_i, sin_i)
    gadf = np.outer(sin_i, cos_i) - np.outer(cos_i, sin_i)
    return (gasf.astype(np.float32), gadf.astype(np.float32))

def _spectrogram_img(x: np.ndarray, size: int, fs: float=1000.0) -> np.ndarray:
    nper = max(16, len(x) // 16)
    f, t, sxx = spectrogram(x.astype(np.float64), fs=fs, nperseg=nper, noverlap=nper // 2)
    sxx = np.log1p(sxx)
    img = _resize2d(sxx, size, size)
    return img.astype(np.float32)

def _resize2d(a: np.ndarray, h: int, w: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.shape[0] < 2:
        a = np.repeat(a, 2, axis=0)
    if a.shape[1] < 2:
        a = np.repeat(a, 2, axis=1)
    yi = np.linspace(0, a.shape[0] - 1, h)
    xi = np.linspace(0, a.shape[1] - 1, w)
    y0 = np.floor(yi).astype(int)
    y1 = np.clip(y0 + 1, 0, a.shape[0] - 1)
    x0 = np.floor(xi).astype(int)
    x1 = np.clip(x0 + 1, 0, a.shape[1] - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bot = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy

def _mtf(x: np.ndarray, size: int, n_bins: int=8) -> np.ndarray:
    xs = resample(x, size).astype(np.float64)
    edges = np.quantile(xs, np.linspace(0, 1, n_bins + 1)[1:-1])
    q = np.digitize(xs, edges)
    W = np.zeros((n_bins, n_bins))
    for a, b in zip(q[:-1], q[1:]):
        W[a, b] += 1.0
    row = W.sum(1, keepdims=True)
    W = np.divide(W, row, out=np.zeros_like(W), where=row > 0)
    mtf = W[q][:, q]
    return mtf.astype(np.float32)

def _recurrence(x: np.ndarray, size: int) -> np.ndarray:
    xs = resample(x, size).astype(np.float64)
    xs = _minmax(xs)
    d = np.abs(xs[:, None] - xs[None, :])
    return np.exp(-d * 4.0).astype(np.float32)

def signal_to_vision(x: np.ndarray, size: int=32, fs: float=1000.0, mode: str=None) -> np.ndarray:
    mode = mode or os.environ.get('PPG_VISION_MODE', 'default')
    x = np.asarray(x, dtype=np.float64).ravel()
    gasf, gadf = _gaf(x, size)
    if mode == 'mtf':
        c1 = _mtf(x, size)
        c2 = _recurrence(x, size)
        img = np.stack([_minmax(gasf), _minmax(c1), _minmax(c2)], axis=-1)
    else:
        spec = _spectrogram_img(x, size, fs=fs)
        img = np.stack([_minmax(gasf), _minmax(gadf), _minmax(spec)], axis=-1)
    return img.astype(np.float32)

def batch_signal_to_vision(signals: np.ndarray, size: int=32, fs: float=1000.0) -> np.ndarray:
    return np.stack([signal_to_vision(s, size=size, fs=fs) for s in signals], axis=0)
