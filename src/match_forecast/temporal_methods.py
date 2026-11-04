"""Поправки вероятностей и интервалы по прошлым прогнозам."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

CALIBRATORS = ("none", "temperature", "temperature_shrunk", "affine", "isotonic")


def normalize(p):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return p / p.sum(axis=1, keepdims=True)


def log_loss(y, p):
    return -np.log(normalize(p)[np.arange(len(y)), np.asarray(y, int)])


def pav(x, y):
    order = np.argsort(x, kind="stable")
    x, y = np.asarray(x)[order], np.asarray(y, float)[order]
    values, starts, counts = np.unique(x, return_index=True, return_counts=True)
    sums = np.add.reduceat(y, starts)
    blocks = []
    for i, (total, count) in enumerate(zip(sums, counts)):
        blocks.append([i, i, float(total), int(count)])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    fitted = np.empty(len(values))
    for left, right, total, count in blocks:
        fitted[left:right + 1] = total / count
    return {"x": values.tolist(), "y": fitted.tolist()}


def fit_calibrator(p, y, method):
    p, y = normalize(p), np.asarray(y, int)
    if len(p) != len(y) or len(y) < 30 or set(np.unique(y)) != set(range(p.shape[1])):
        raise ValueError("Calibration requires every class and at least 30 observations")
    state = {"method": method, "n": len(y), "classes": p.shape[1]}
    if method == "none":
        return state
    if method == "isotonic":
        classes = [1] if p.shape[1] == 2 else range(p.shape[1])
        state["curves"] = {str(k): pav(p[:, k], y == k) for k in classes}
        return state
    if method not in CALIBRATORS:
        raise ValueError(method)
    logits = np.log(p)
    n_bias = p.shape[1] - 1 if method == "affine" else 0

    def objective(parameters):
        inverse_t = np.exp(parameters[0])
        bias = np.r_[parameters[1:], 0.] if n_bias else np.zeros(p.shape[1])
        adjusted = softmax(logits * inverse_t + bias, axis=1)
        penalty = .01 * np.square(parameters).mean() if method in ("affine", "temperature_shrunk") else 0.
        return float(log_loss(y, adjusted).mean() + penalty)

    result = minimize(objective, np.zeros(1 + n_bias), method="L-BFGS-B",
                      bounds=[(-1.5, 1.5)] + [(-3, 3)] * n_bias)
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Calibration failed: {method}, {result.message}")
    state["inverse_temperature"] = float(np.exp(result.x[0]))
    state["bias"] = np.r_[result.x[1:], 0.].tolist() if n_bias else [0.] * p.shape[1]
    return state


def apply_calibrator(p, state):
    if state["method"] == "none":
        return np.asarray(p, float).copy()
    p = normalize(p)
    if state["method"] == "isotonic":
        if p.shape[1] == 2:
            curve = state["curves"]["1"]
            positive = np.interp(p[:, 1], curve["x"], curve["y"])
            return normalize(np.column_stack([1 - positive, positive]))
        corrected = np.column_stack([
            np.interp(p[:, k], state["curves"][str(k)]["x"], state["curves"][str(k)]["y"])
            for k in range(p.shape[1])
        ])
        return normalize(corrected)
    return softmax(np.log(p) * state["inverse_temperature"] + state["bias"], axis=1)


def history_before(frame, period, recent=None, embargo_days=0):
    history = frame.loc[frame.period.astype(str) < str(period)].copy()
    if embargo_days:
        import pandas as pd
        cutoff = pd.Timestamp(str(period) + "-01") - pd.Timedelta(days=embargo_days)
        history = history.loc[pd.to_datetime(history.date) < cutoff]
    if recent is not None:
        periods = sorted(history.period.astype(str).unique())[-int(recent):]
        history = history.loc[history.period.astype(str).isin(periods)]
    return history


def joint_probabilities(frame):
    observed = frame.pred_observed.to_numpy(float)
    eligible = frame.pred_eligible.to_numpy(float)
    if np.any(eligible > observed + 1e-6):
        raise ValueError("Eligible probability exceeds observed probability")
    return normalize(np.column_stack([1 - observed, observed - eligible, eligible]))


def joint_targets(frame):
    return np.where(frame.target_observed == 0, 0, np.where(frame.target_eligible == 1, 2, 1))


FLOORS = {"minutes": 300., "npxg90": .05, "xa90": .03}


def interval_scale(prediction, target, method):
    if method == "absolute":
        return np.ones(len(prediction))
    if method == "relative":
        return np.maximum(np.asarray(prediction, float), FLOORS[target])
    raise ValueError(method)


def fit_interval(history, target, method, alpha=.1):
    actual = history[f"target_{target}"].to_numpy(float)
    predicted = history[f"pred_{target}"].to_numpy(float)
    mask = np.isfinite(actual)
    residuals = np.abs(actual[mask] - predicted[mask]) / interval_scale(predicted[mask], target, method)
    if len(residuals) < 30:
        raise ValueError("Too few interval calibration residuals")
    rank = min(int(np.ceil((len(residuals) + 1) * (1 - alpha))), len(residuals))
    return {"method": method, "target": target, "n": len(residuals), "alpha": alpha,
            "radius": float(np.partition(residuals, rank - 1)[rank - 1])}


def apply_interval(prediction, state):
    radius = state["radius"] * interval_scale(prediction, state["target"], state["method"])
    return np.maximum(np.asarray(prediction) - radius, 0), np.asarray(prediction) + radius


def interval_metrics(actual, lower, upper, alpha=.1):
    actual = np.asarray(actual, float)
    valid = np.isfinite(actual)
    y, lo, hi = actual[valid], np.asarray(lower)[valid], np.asarray(upper)[valid]
    width = hi - lo
    score = width + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return {"n": len(y), "coverage": float(((lo <= y) & (y <= hi)).mean()),
            "mean_width": float(width.mean()), "interval_score": float(score.mean())}

