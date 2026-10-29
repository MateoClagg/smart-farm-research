"""
Cook Agronomy Farm (WSU, Pullman WA): 42 sites, daily, 2011-2012, Decagon 5TE sensors at
0.3/0.6/0.9/1.2/1.5 m (Port1..Port5) measuring volumetric water, soil temperature and bulk EC.

Used to test the NPK-probe idea: low-cost NPK probes convert one conductivity signal into N, P and K
at fixed ratios, so forecasting the probe's NPK reduces to forecasting EC.
"""
import numpy as np
import pandas as pd

from .data import load_cookfarm


def daily_panel(port=1):
    cf = load_cookfarm()
    r, w, bd = cf["readings"], cf["weather"], cf["bdensity"]
    days = pd.date_range(w.Date.min(), w.Date.max(), freq="D")
    out = []
    for sid, g in r.groupby("SOURCEID"):
        g = g.set_index("Date").reindex(days).rename_axis("Date").reset_index()
        g["SOURCEID"] = sid
        out.append(g)
    p = pd.concat(out, ignore_index=True).rename(
        columns={f"Port{port}VW": "vw", f"Port{port}C": "st", f"Port{port}EC": "ec"})
    p = p[["SOURCEID", "Date", "vw", "st", "ec"]].merge(
        w.rename(columns={"Precip_wrcc": "precip", "MaxT_wrcc": "tmax", "MinT_wrcc": "tmin"}), on="Date")
    b = bd[(bd.UHDICM == 20) & (bd.LHDICM == 40)][["SOURCEID", "BLD"]].copy()
    b["SOURCEID"] = b.SOURCEID.astype(str)
    p = p.merge(b.rename(columns={"BLD": "bd"}), on="SOURCEID", how="left")
    p["ec_uScm"] = p.ec * 1000  # dS/m -> uS/cm
    p["year"] = p.Date.dt.year
    g = p.groupby("SOURCEID", group_keys=False)
    for lag in (1, 2):
        p[f"vw_l{lag}"], p[f"ec_l{lag}"] = g.vw.shift(lag), g.ec_uScm.shift(lag)
    p["st_l1"] = g.st.shift(1)
    p["precip_3d"] = g.precip.transform(lambda s: s.rolling(3, min_periods=1).sum())
    doy = p.Date.dt.dayofyear
    p["doy_sin"], p["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for h in (1, 2, 3):
        p[f"fc_precip_{h}d"] = sum(g.precip.shift(-k) for k in range(1, h + 1))
        p[f"fc_tmax_{h}d"] = sum(g.tmax.shift(-k) for k in range(1, h + 1)) / h
        p[f"ec_t{h}"] = g.ec_uScm.shift(-h)
        p[f"vw_t{h}"] = g.vw.shift(-h)
    return p


EC_FEATURES = ["vw", "vw_l1", "vw_l2", "st", "st_l1", "ec_uScm", "ec_l1", "ec_l2", "precip", "precip_3d",
               "tmax", "tmin", "doy_sin", "doy_cos", "bd"]
