import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, accuracy_score, roc_auc_score, precision_score, recall_score

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = np.clip(np.abs(y_true), 1e-06, None)
    return {'MAE': float(mean_absolute_error(y_true, y_pred)), 'MAPE': float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0), 'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))), 'R2': float(r2_score(y_true, y_pred))}

def best_threshold(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return 0.5
    cands = np.unique(np.concatenate([[0.5], np.quantile(y_prob, np.linspace(0.05, 0.95, 19))]))
    best, bthr = (-1.0, 0.5)
    for t in cands:
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best:
            best, bthr = (f, float(t))
    return bthr

def classification_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= thr).astype(int)
    out = {'Precision': float(precision_score(y_true, y_pred, zero_division=0)), 'Recall': float(recall_score(y_true, y_pred, zero_division=0)), 'F1': float(f1_score(y_true, y_pred, zero_division=0)), 'Acc': float(accuracy_score(y_true, y_pred))}
    try:
        out['AUC'] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan')
    except Exception:
        out['AUC'] = float('nan')
    return out

def rankme(features, eps=1e-07):
    Z = np.asarray(features, dtype=np.float64)
    Z = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / (s.sum() + eps) + eps
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))

def alpha_req(features, eps=1e-12):
    Z = np.asarray(features, dtype=np.float64)
    Z = Z - Z.mean(0, keepdims=True)
    cov = np.cov(Z, rowvar=False)
    eig = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
    eig = eig[eig > eps]
    if len(eig) < 3:
        return float('nan')
    rank = np.arange(1, len(eig) + 1)
    coeffs = np.polyfit(np.log(rank), np.log(eig), 1)
    return float(-coeffs[0])
