"""Tabular models. Learned models predict the 24 h CHANGE in the target; we add the current value back."""
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED

GBM_DEFAULT = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


class Linear:
    """Ridge regression with median imputation and scaling."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.med_ = X.median()
        self.m_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(X.fillna(self.med_), y)
        return self

    def predict(self, X):
        return self.m_.predict(X.fillna(self.med_))


def make_model(kind, params=None, device="cpu"):
    params = dict(params or {})
    if kind == "linear":
        return Linear(**params)
    if kind == "gbm":
        p = {**GBM_DEFAULT, **params}
        return lgb.LGBMRegressor(random_state=SEED, verbose=-1, **p)
    if kind == "xgb":  # GPU-friendly alternative: device="cuda" on Colab / RTX
        import xgboost as xgb
        p = dict(n_estimators=800, learning_rate=0.03, max_depth=8, subsample=0.8,
                 colsample_bytree=0.8, tree_method="hist", device=device)
        p.update(params)
        return xgb.XGBRegressor(random_state=SEED, **p)
    raise ValueError(kind)


def fit_predict_delta(kind, Xtr, current_tr, target_tr, Xte, current_te, params=None, device="cpu",
                      clip_min=None):
    m = make_model(kind, params, device).fit(Xtr, target_tr - current_tr)
    pred = np.asarray(current_te) + m.predict(Xte)
    return (np.clip(pred, clip_min, None) if clip_min is not None else pred), m


def fit_predict_level(kind, Xtr, ytr, Xte, params=None, device="cpu", clip_min=0.0):
    m = make_model(kind, params, device).fit(Xtr, ytr)
    return np.clip(m.predict(Xte), clip_min, None), m
