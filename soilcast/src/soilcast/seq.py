"""
Sequence model (LSTM) for 24 h soil moisture forecasting.

Inputs per sample (issue time t):
  past   : hourly sensors over [t-P+1, t]   -> sm, st, air_t, rh, log1p(precip), dsm_1h, sm_missing
  future : hourly forecast over [t+1, t+24] -> log1p(precip), air_t, rh   (perfect or NBM lead-day-1)
  static : SSURGO sand, silt, clay, awc, drainage, bulk density
Output: soil moisture change at +1..+24 h (percentage points). Loss: Huber, NaN targets masked.

Runs on CPU for smoke tests; set device="cuda" on Colab A100 / RTX 4080 Super.
"""
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import FROZEN_C, SEED, TEST_YEARS, TRAIN_YEARS
from .features import SOIL, hourly_panel, soil_table

PAST = ["sm", "st", "air_t", "rh", "precip_log", "dsm_1h", "sm_missing"]
FUT = ["f_precip_log", "f_air_t", "f_rh"]
H = 24


def hourly_for_seq(uscrn, soils, depth, forecasts=None):
    """Hourly frame with sensors, soils and lead-day-1 forecast columns (f_*)."""
    h = hourly_panel(uscrn, depth).merge(soil_table(soils, depth), on="station", how="left")
    h["precip_log"] = np.log1p(h.precip.clip(lower=0))
    h["dsm_1h"] = h.groupby("station").sm.diff()
    h["sm_missing"] = h.sm.isna().astype(float)
    if forecasts is None:
        f = h[["station", "time_utc", "precip", "air_t", "rh"]]
    else:
        f = forecasts[forecasts.lead_day == 1][["station", "time_utc", "precip", "air_t", "rh"]]
    f = f.rename(columns={"precip": "f_precip", "air_t": "f_air_t", "rh": "f_rh"})
    h = h.merge(f, on=["station", "time_utc"], how="left")
    h["f_precip_log"] = np.log1p(h.f_precip.clip(lower=0))
    h["year"] = h.time_utc.dt.year
    return h


class Normalizer:
    def fit(self, h, cols):
        self.cols = cols
        self.mu = h[cols].mean().to_numpy(np.float32)
        self.sd = (h[cols].std().to_numpy(np.float32) + 1e-6)
        return self

    def __call__(self, x):
        return (x - self.mu) / self.sd


class SeqData(Dataset):
    """Slices windows on the fly from per-station arrays (memory-light)."""

    def __init__(self, h, years, past_len, norm_p, norm_f, norm_s, step_h=1, require_fut=True):
        self.past_len = past_len
        self.items, self.P, self.F, self.S, self.Y, self.keys = [], [], [], [], [], []
        for si, (st, g) in enumerate(h.groupby("station", sort=True)):
            g = g.sort_values("time_utc").reset_index(drop=True)
            sm, stemp = g.sm.to_numpy(), g.st.to_numpy()
            P = np.nan_to_num(norm_p(g[PAST].to_numpy(np.float32)))
            F = g[FUT].to_numpy(np.float32)
            fut_ok = ~np.isnan(F).any(1)
            F = np.nan_to_num(norm_f(F))
            S = np.nan_to_num(norm_s(g[SOIL].iloc[:1].to_numpy(np.float32)))[0]
            n = len(g)
            Y = np.full((n, H), np.nan, np.float32)
            for k in range(1, H + 1):
                Y[:n - k, k - 1] = 100 * (sm[k:] - sm[:-k])           # change in pp
            valid_sm = ~np.isnan(sm)
            past_cov = pd.Series(valid_sm.astype(float)).rolling(past_len, min_periods=1).mean().to_numpy()
            fut_cov = pd.Series(fut_ok.astype(float))[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-1).to_numpy()
            t = np.arange(n)
            ok = (t >= past_len) & (t < n - H) & valid_sm & (past_cov >= 0.8) & g.year.isin(years).to_numpy()
            ok &= np.r_[valid_sm[H:], np.zeros(H, bool)] & (stemp > FROZEN_C)
            ok &= np.r_[stemp[H:] > FROZEN_C, np.zeros(H, bool)]
            ok &= (g.time_utc.dt.hour.to_numpy() % step_h == 0)
            if require_fut:
                ok &= np.nan_to_num(fut_cov) >= H
            idx = np.where(ok)[0]
            self.P.append(torch.from_numpy(P)); self.F.append(torch.from_numpy(F))
            self.S.append(torch.from_numpy(S)); self.Y.append(torch.from_numpy(Y))
            self.items += [(si, i) for i in idx]
            self.keys.append(pd.DataFrame({"station": st, "time_utc": g.time_utc.to_numpy()[idx],
                                           "sm": sm[idx]}))
        self.keys = pd.concat(self.keys, ignore_index=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        si, t = self.items[i]
        return (self.P[si][t - self.past_len + 1:t + 1], self.F[si][t + 1:t + 1 + H],
                self.S[si], self.Y[si][t])


class SoilLSTM(nn.Module):
    def __init__(self, n_past=len(PAST), n_fut=len(FUT), n_static=len(SOIL), hidden=128, fut_hidden=64,
                 layers=2, dropout=0.2, static_dim=16, head=128):
        super().__init__()
        self.past = nn.LSTM(n_past, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.fut = nn.LSTM(n_fut, fut_hidden, 1, batch_first=True)
        self.static = nn.Sequential(nn.Linear(n_static, static_dim), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + fut_hidden + static_dim, head), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head, H))

    def forward(self, p, f, s):
        _, (hp, _) = self.past(p)
        _, (hf, _) = self.fut(f)
        return self.head(torch.cat([hp[-1], hf[-1], self.static(s)], dim=1))


def _masked_huber(pred, y, delta):
    m = ~torch.isnan(y)
    return nn.functional.huber_loss(pred[m], torch.nan_to_num(y)[m], delta=delta)


def train_lstm(h_train, h_eval_sets, past_len=72, hidden=128, fut_hidden=64, layers=2, dropout=0.2,
               lr=2e-3, weight_decay=1e-4, batch_size=512, epochs=20, patience=4, huber_delta=1.0,
               train_step_h=1, eval_step_h=3, max_train=None, device=None, num_workers=0, verbose=True,
               valid_year=2023):
    """Train on TRAIN_YEARS except `valid_year` (early stopping), return model + predictions per eval set.

    h_eval_sets: dict name -> hourly frame (e.g. {"perfect": h, "nbm": h_nbm}) evaluated on TEST_YEARS.
    """
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fit_years = [y for y in TRAIN_YEARS if y != valid_year]
    hf = h_train[h_train.year.isin(fit_years)]
    norm_p = Normalizer().fit(hf, PAST)
    norm_f = Normalizer().fit(hf, FUT)
    norm_s = Normalizer().fit(hf.drop_duplicates("station"), SOIL)

    tr = SeqData(h_train, fit_years, past_len, norm_p, norm_f, norm_s, step_h=train_step_h)
    va = SeqData(h_train, [valid_year], past_len, norm_p, norm_f, norm_s, step_h=eval_step_h)
    if max_train and len(tr) > max_train:
        rng = np.random.default_rng(SEED)
        tr.items = [tr.items[i] for i in rng.choice(len(tr.items), max_train, replace=False)]
    pin = device == "cuda"
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(va, batch_size=4 * batch_size, num_workers=num_workers, pin_memory=pin)

    model = SoilLSTM(hidden=hidden, fut_hidden=fut_hidden, layers=layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(dl_tr)))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best, best_state, bad, hist = math.inf, None, 0, []

    def run_eval(dl):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            for p, f, s, y in dl:
                out.append(model(p.to(device), f.to(device), s.to(device)).float().cpu())
        return torch.cat(out).numpy()

    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = n = 0
        for p, f, s, y in dl_tr:
            p, f, s, y = p.to(device), f.to(device), s.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = _masked_huber(model(p, f, s).float(), y, huber_delta)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        pv = run_eval(dl_va)
        yv = np.stack([va.Y[si][t].numpy() for si, t in va.items])[:, H - 1]
        v_rmse = float(np.sqrt(np.nanmean((pv[:, H - 1] - yv) ** 2)))
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "valid_rmse_pp_24h": v_rmse, "sec": time.time() - t0})
        if verbose:
            print(f"epoch {ep + 1:2d}  train {tot / n:.4f}  valid RMSE(24h) {v_rmse:.3f} pp  {time.time() - t0:.0f}s")
        if v_rmse < best - 1e-4:
            best, bad = v_rmse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    preds = {}
    for name, h_eval in h_eval_sets.items():
        te = SeqData(h_eval, list(TEST_YEARS), past_len, norm_p, norm_f, norm_s, step_h=eval_step_h)
        pt = run_eval(DataLoader(te, batch_size=4 * batch_size, num_workers=num_workers))
        k = te.keys.copy()
        k["pred_sm_t24"] = k.sm + pt[:, H - 1] / 100
        preds[name] = k
    return model, pd.DataFrame(hist), preds, len(tr)
