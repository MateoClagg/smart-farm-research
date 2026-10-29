"""Train the LSTM on GPU and evaluate with perfect and NBM forecasts.

  python scripts/train_lstm.py --depth 5 --epochs 30 --hidden 128 --past-len 72 --device cuda
  python scripts/train_lstm.py --sweep 20 --device cuda      # random search over LSTM hyperparameters

Writes results/lstm/<run>.json and predictions parquet.
"""
import argparse
import os
import json
import random

import pandas as pd

from soilcast import data
from soilcast.config import RESULTS, SEED
from soilcast.evaluate import sm_metrics
from soilcast.seq import hourly_for_seq, train_lstm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--past-len", type=int, default=72)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    ap.add_argument("--sweep", type=int, default=0, help="number of random-search configs (0 = single run)")
    a = ap.parse_args()

    u, soils, nbm = data.load_uscrn(), data.load_soils(), data.load_forecasts()
    h = hourly_for_seq(u, soils, a.depth)
    h_nbm = hourly_for_seq(u[u.time_utc >= "2024-09-01"], soils, a.depth, forecasts=nbm)
    out = RESULTS / "lstm"
    out.mkdir(parents=True, exist_ok=True)

    base = dict(past_len=a.past_len, hidden=a.hidden, layers=a.layers, lr=a.lr, dropout=a.dropout)
    configs = [base]
    if a.sweep:
        rng = random.Random(SEED)
        configs = [dict(past_len=rng.choice([24, 48, 72, 168]), hidden=rng.choice([64, 128, 256]),
                        layers=rng.choice([1, 2, 3]), lr=10 ** rng.uniform(-3.5, -2.3),
                        dropout=rng.choice([0.0, 0.1, 0.2, 0.3])) for _ in range(a.sweep)]

    for i, cfg in enumerate(configs):
        model, hist, preds, n_train = train_lstm(h, {"perfect": h, "nbm": h_nbm}, epochs=a.epochs,
                                                 batch_size=a.batch_size, device=a.device,
                                                 num_workers=a.workers, **cfg)
        res = {}
        for name, p in preds.items():
            target = h[["station", "time_utc", "sm"]].copy()
            target["sm_t24"] = target.groupby("station").sm.shift(-24)
            target["fc_precip_24h"] = h.groupby("station").precip.transform(
                lambda s: s[::-1].rolling(24, min_periods=12).sum()[::-1].shift(-1))
            m = p.merge(target[["station", "time_utc", "sm_t24", "fc_precip_24h"]], on=["station", "time_utc"])
            if name == "nbm":
                m = m[m.time_utc >= "2025-01-01"]
            res[name] = sm_metrics(m.sm_t24, {"persistence": m.sm, "lstm": m.pred_sm_t24}, m.sm,
                                   m.fc_precip_24h).to_dict("records")
            p.to_parquet(out / f"lstm_{a.depth}cm_run{i}_{name}_preds.parquet")
        rec = {"config": cfg, "n_train": n_train, "best_valid_rmse": float(hist.valid_rmse_pp_24h.min()),
               "history": hist.to_dict("records"), "test": res}
        (out / f"lstm_{a.depth}cm_run{i}.json").write_text(json.dumps(rec, indent=2, default=float))
        print(json.dumps({"run": i, **cfg, "valid": rec["best_valid_rmse"],
                          "test_perfect": res["perfect"][1]["RMSE_pp"], "test_nbm": res["nbm"][1]["RMSE_pp"]}))


if __name__ == "__main__":   # required on Windows when --workers > 0
    main()
