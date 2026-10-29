"""Splits and metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import CV_VALID_YEARS, EVENT_MM, TEST_YEARS, TRAIN_YEARS


def rmse(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def time_split(d):
    return d[d.year.isin(TRAIN_YEARS)], d[d.year.isin(TEST_YEARS)]


def loso_splits(d):
    """Leave one station out AND future years: train on other stations' train years, test on this station's test years."""
    tr, te = time_split(d)
    for s in sorted(d.station.unique()):
        yield s, tr[tr.station != s], te[te.station == s]


def cv_folds(d, valid_years=CV_VALID_YEARS):
    """Expanding window inside the training years: fit on years < Y, validate on Y."""
    tr = d[d.year.isin(TRAIN_YEARS)]
    for y in valid_years:
        yield y, tr[tr.year < y], tr[tr.year == y]


def sm_metrics(y_true, preds, current, fc_rain, scale=100.0):
    """Soil moisture metrics in percentage points (pp). preds: dict name -> prediction."""
    y_true, current = np.asarray(y_true), np.asarray(current)
    ev = np.asarray(fc_rain) >= EVENT_MM
    pr = rmse(y_true, current)
    rows = []
    for name, yhat in preds.items():
        yhat = np.asarray(yhat)
        rows.append({"model": name,
                     "RMSE_pp": scale * rmse(y_true, yhat),
                     "MAE_pp": scale * float(np.mean(np.abs(y_true - yhat))),
                     "skill_vs_persistence": 1 - rmse(y_true, yhat) / pr,
                     "RMSE_pp_rain_events": scale * rmse(y_true[ev], yhat[ev]) if ev.any() else np.nan,
                     "n": len(y_true), "n_rain_events": int(ev.sum())})
    return pd.DataFrame(rows)


def nlri_metrics(y_true, preds, past24, risk_thresh=0.05):
    """Nitrate loss index metrics. 'onset' = no risk in the past 24 h (persistence can't see these)."""
    y_true = np.asarray(y_true)
    risk = (y_true > risk_thresh).astype(int)
    onset = np.asarray(past24) == 0
    rows = []
    for name, yhat in preds.items():
        yhat = np.asarray(yhat)
        r = {"model": name, "RMSE": rmse(y_true, yhat), "MAE": float(np.mean(np.abs(y_true - yhat))),
             "share_with_risk": float(risk.mean()), "n": len(y_true)}
        if 0 < risk.sum() < len(risk):
            r["AUC_any_risk"] = roc_auc_score(risk, yhat)
        if 0 < risk[onset].sum() < onset.sum():
            r["AUC_new_risk"] = roc_auc_score(risk[onset], yhat[onset])
            r["n_new_risk"] = int(risk[onset].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_rmse_ci(y, yhat, groups, n=500, seed=0):
    """95% CI for RMSE by resampling whole groups (e.g. station-months) to respect autocorrelation."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"se": (np.asarray(y) - np.asarray(yhat)) ** 2, "g": np.asarray(groups)})
    by = df.groupby("g").se.agg(["sum", "count"])
    keys = by.index.to_numpy()
    stats = []
    for _ in range(n):
        pick = by.loc[rng.choice(keys, len(keys), replace=True)]
        stats.append(np.sqrt(pick["sum"].sum() / pick["count"].sum()))
    return np.percentile(stats, [2.5, 97.5])
