"""Optuna tuning with expanding-window CV inside the training years (test years never touched)."""
import numpy as np
import optuna

from .evaluate import cv_folds, rmse
from .models import make_model


def gbm_space(trial):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 3000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 256, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 500, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        # Heavy-tailed target (rain jumps): let the tuner choose squared vs Huber loss
        objective=trial.suggest_categorical("objective", ["regression", "huber"]),
    )
    if p["objective"] == "huber":
        # Huber threshold in target units (m3/m3 change); LightGBM's default 0.9 would equal squared loss here
        p["alpha"] = trial.suggest_float("huber_delta", 0.002, 0.05, log=True)
    return p


def xgb_space(trial):
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 4000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        objective=trial.suggest_categorical("objective", ["reg:squarederror", "reg:pseudohubererror"]),
    )
    if p["objective"] == "reg:pseudohubererror":
        p["huber_slope"] = trial.suggest_float("huber_delta", 0.002, 0.05, log=True)
    return p


def params_from_trial(best_params, kind="gbm"):
    """Map a study's best_params back to model kwargs."""
    p = dict(best_params)
    delta = p.pop("huber_delta", None)
    if delta is not None:
        p["alpha" if kind == "gbm" else "huber_slope"] = delta
    if kind == "gbm":
        p["subsample_freq"] = 1
    return p


def make_objective(d, features, target, current=None, kind="gbm", device="cpu", max_rows=None, seed=0):
    """Mean RMSE over CV folds. If `current` is given the model learns the change (target - current)."""
    folds = list(cv_folds(d))
    space = gbm_space if kind == "gbm" else xgb_space
    rng = np.random.default_rng(seed)

    def objective(trial):
        params = space(trial)
        scores = []
        for i, (year, tr, va) in enumerate(folds):
            if max_rows and len(tr) > max_rows:
                tr = tr.iloc[np.sort(rng.choice(len(tr), max_rows, replace=False))]
            ytr = tr[target] - tr[current] if current else tr[target]
            m = make_model(kind, params, device).fit(tr[features], ytr)
            pred = m.predict(va[features]) + (va[current].to_numpy() if current else 0)
            scores.append(rmse(va[target], pred))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def run_study(d, features, target, current=None, kind="gbm", n_trials=50, device="cpu",
              storage=None, study_name=None, max_rows=None, timeout=None):
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage,
                                load_if_exists=bool(storage),
                                sampler=optuna.samplers.TPESampler(seed=7),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(make_objective(d, features, target, current, kind, device, max_rows),
                   n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return study
