"""Extensive hyperparameter tuning (Optuna, expanding-window CV over 2020-2023; test years untouched).

Examples
  # LightGBM on CPU, 300 trials, resumable study stored in SQLite
  python scripts/tune_gbm.py --target sm --depth 5 --trials 300 --storage sqlite:///results/tuning.db

  # XGBoost on GPU (Colab A100 / RTX 4080 Super)
  python scripts/tune_gbm.py --kind xgb --device cuda --trials 500 --storage sqlite:///results/tuning.db

  # Nitrate loss index (10 cm)
  python scripts/tune_gbm.py --target nlri --depth 10 --trials 200

After tuning, the best params are refit on all training years and scored once on 2024-2025.
"""
import argparse
import json

import numpy as np
import pandas as pd

from soilcast.config import PROCESSED, RESULTS
from soilcast.evaluate import nlri_metrics, sm_metrics, time_split
from soilcast.features import FC72, FEATURE_SETS, NLRI_EXTRA, SENSORS, SOIL, nlri_valid, sm_valid
from soilcast.models import make_model
from soilcast.tuning import params_from_trial, run_study

ap = argparse.ArgumentParser()
ap.add_argument("--target", choices=["sm", "nlri"], default="sm")
ap.add_argument("--depth", type=int, default=5)
ap.add_argument("--features", default="v2_engineered", choices=list(FEATURE_SETS))
ap.add_argument("--kind", choices=["gbm", "xgb"], default="gbm")
ap.add_argument("--device", default="cpu")
ap.add_argument("--trials", type=int, default=100)
ap.add_argument("--timeout", type=int, default=None, help="seconds")
ap.add_argument("--max-rows", type=int, default=None, help="subsample training rows per fold")
ap.add_argument("--storage", default=None)
a = ap.parse_args()

d = pd.read_parquet(PROCESSED / f"features_{a.depth}cm_perfect.parquet")
if a.target == "sm":
    d, feats, target, current = d[sm_valid(d)], FEATURE_SETS[a.features], "sm_t24", "sm"
else:
    d, target, current = d[nlri_valid(d)], "nlri_72h", None
    feats = SENSORS + SOIL + NLRI_EXTRA + FC72

name = f"{a.target}_{a.depth}cm_{a.kind}_{a.features}"
study = run_study(d, feats, target, current, a.kind, a.trials, a.device, a.storage, name, a.max_rows, a.timeout)
best = params_from_trial(study.best_params, a.kind)
print("best CV RMSE:", study.best_value, "\nparams:", json.dumps(best, indent=2))

tr, te = time_split(d)
m = make_model(a.kind, best, a.device)
if a.target == "sm":
    m.fit(tr[feats], tr[target] - tr[current])
    pred = te[current].to_numpy() + m.predict(te[feats])
    res = sm_metrics(te[target], {"persistence": te[current], f"tuned {a.kind}": pred}, te[current], te.fc_precip_24h)
else:
    m.fit(tr[feats], tr[target])
    pred = np.clip(m.predict(te[feats]), 0, None)
    res = nlri_metrics(te[target], {"persistence": 72 * te.nl_rate_h, f"tuned {a.kind}": pred}, te.nl_past24)
print(res.round(4).to_string(index=False))

out = RESULTS / "tuning"
out.mkdir(parents=True, exist_ok=True)
study.trials_dataframe().to_csv(out / f"{name}_trials.csv", index=False)
(out / f"{name}_best.json").write_text(json.dumps({"cv_rmse": study.best_value, "params": best,
                                                   "test": res.to_dict("records")}, indent=2))
