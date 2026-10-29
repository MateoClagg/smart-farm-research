"""Generate the notebooks (run from repo root: python tools/make_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = """# Setup: works locally (pip install -e . from the repo root) and on Google Colab
import sys, warnings
warnings.filterwarnings("ignore")
if "google.colab" in sys.modules:
    # Upload soilcast.zip to Google Drive (MyDrive) first
    from google.colab import drive
    drive.mount("/content/drive")
    !unzip -q -o /content/drive/MyDrive/soilcast.zip -d /content
    %cd /content/soilcast
    !pip install -q -e ".[gpu-trees]"
    sys.path.insert(0, "/content/soilcast/src")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from soilcast.config import PROCESSED, RAW, RESULTS
from soilcast.plotting import set_style, model_color, SERIES, MUTED
set_style()
pd.set_option("display.width", 160, "display.max_columns", 30, "display.precision", 3)
RESULTS.mkdir(exist_ok=True)"""


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    n.cells = out
    nbf.write(n, NB / name)
    print("wrote", name)


md = lambda s: ("md", s)
code = lambda s: ("code", s)

# =============================================================================================== 01
nb("01_data_overview.ipynb", [
    md("""# 01 · Data overview

What we train on while the Jolly Lane prototype collects data:

| Source | Role in her design |
|---|---|
| **NOAA USCRN hourly**, 10 northern Great Plains stations, 2016 to 2025 | Sensor inputs: soil moisture and soil temp (5/10/20 cm), air temp, RH, rain |
| **USDA SSURGO** (Soil Data Access API) at each station | Soil properties: sand/silt/clay, AWC, drainage class, bulk density |
| **NOAA National Blend of Models (NBM)**, archived via Open-Meteo Previous Runs, Oct 2024 onward | Real weather forecasts at 24/48/72 h lead, for the realistic test |
| **Cook Agronomy Farm** (WSU), daily, 2011 to 2012 | EC check for the NPK idea (notebook 06) |

All times are UTC."""),
    code(SETUP),
    code("""from soilcast import data
u = data.load_uscrn()
soils = data.load_soils()
nbm = data.load_forecasts()
coords = data.station_coords(u)
print(u.shape, "hourly rows;", u.station.nunique(), "stations;", u.time_utc.min(), "to", u.time_utc.max())"""),
    md("## Stations and their soils (SSURGO at 5 cm)"),
    code("""s5 = soils[soils.depth_cm == 5].set_index("station")
cov = u.groupby("station").agg(sm5_coverage=("SOIL_MOISTURE_5", lambda s: s.notna().mean()),
                               rain_mm_per_yr=("P_CALC", lambda s: s.sum() / 10),
                               mean_sm5=("SOIL_MOISTURE_5", "mean"))
tbl = coords.join(s5[["compname", "sandtotal_r", "claytotal_r", "awc_r", "dbthirdbar_r", "drainagecl"]]).join(cov)
tbl.round(3)"""),
    code("""fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(tbl.LONGITUDE, tbl.LATITUDE, c=tbl.claytotal_r, cmap="Blues", s=90, edgecolor=MUTED, vmin=0)
for st, r in tbl.iterrows():
    ax.annotate(st.replace("_", " "), (r.LONGITUDE, r.LATITUDE), xytext=(5, 4), textcoords="offset points", fontsize=8)
ax.scatter([-103.23], [44.08], marker="*", s=220, color="#eb6834", label="Rapid City / Jolly Lane")
plt.colorbar(sc, label="clay % at 5 cm (SSURGO)")
ax.set(xlabel="longitude", ylabel="latitude", title="USCRN stations used for training")
ax.legend(loc="lower left");"""),
    md("""## When is soil moisture available?
USCRN moisture probes are dielectric sensors; frozen soil reads wrong and is mostly flagged missing. Our models only cover unfrozen soil (roughly April to November)."""),
    code("""m = u.assign(month=u.time_utc.dt.month).groupby(["station", "month"]).SOIL_MOISTURE_5.apply(lambda s: s.notna().mean()).unstack()
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(m.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_yticks(range(len(m)), m.index, fontsize=8); ax.set_xticks(range(12), range(1, 13))
ax.set(xlabel="month", title="Share of hours with a valid 5 cm soil moisture reading"); ax.grid(False)
plt.colorbar(im);"""),
    md("## One growing season up close (SD Buffalo, 2023)"),
    code("""b = u[(u.station == "SD_Buffalo_13_ESE") & (u.time_utc >= "2023-04-15") & (u.time_utc < "2023-10-15")]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
daily = b.set_index("time_utc").P_CALC.resample("D").sum()
a1.bar(daily.index, daily.values, width=0.9, color=SERIES[0]); a1.set(ylabel="rain (mm/day)", title="Rain and soil moisture response")
for i, dep in enumerate((5, 10, 20)):
    a2.plot(b.time_utc, 100 * b[f"SOIL_MOISTURE_{dep}"], lw=1.4, color=SERIES[i + 1], label=f"{dep} cm")
a2.set(ylabel="soil moisture (%)"); a2.legend(ncol=3);"""),
    md("""What to notice:
- Each rain makes a sharp jump, biggest at 5 cm, smaller and later at depth, then a slow curved dry-down. That shape drives the features in notebook 02.
- The daily sawtooth at 10 cm in midsummer: steps down by day, flat at night (plant water uptake follows sunlight; part may be sensor temperature sensitivity).
- Sudden drops with no rain (e.g. 20 cm in mid-September) look like sensor or installation artifacts. A QC step that flags these is worth adding before training on the Jolly Lane probe."""),
    md("""## How good are real forecasts? (NBM vs what actually happened)
`fc_precip_24h` is the rain total over the next 24 h. In the perfect-forecast tables it's observed rain; in the NBM tables it comes from the forecast issued about 24 h earlier."""),
    code("""p = pd.read_parquet(PROCESSED / "features_5cm_perfect.parquet")
n = pd.read_parquet(PROCESSED / "features_5cm_nbm.parquet")
m = p[["station", "time_utc", "fc_precip_24h", "fc_air_t_mean_24h", "fc_rh_mean_24h"]].merge(
    n[["station", "time_utc", "fc_precip_24h", "fc_air_t_mean_24h", "fc_rh_mean_24h"]], on=["station", "time_utc"], suffixes=("_obs", "_nbm")).dropna()
m = m[m.time_utc >= "2025-04-01"]
obs, fc = m.fc_precip_24h_obs, m.fc_precip_24h_nbm
hit, miss = ((obs >= 5) & (fc >= 5)).sum(), ((obs >= 5) & (fc < 5)).sum()
fa = ((obs < 5) & (fc >= 5)).sum()
print(f"24 h rain: correlation {obs.corr(fc):.2f}, MAE {np.mean(np.abs(obs - fc)):.2f} mm")
print(f">= 5 mm events: hit rate (POD) {hit / (hit + miss):.2f}, false alarm ratio {fa / (hit + fa):.2f}, CSI {hit / (hit + miss + fa):.2f}")
print(f"air temp MAE {np.mean(np.abs(m.fc_air_t_mean_24h_obs - m.fc_air_t_mean_24h_nbm)):.2f} C, "
      f"RH MAE {np.mean(np.abs(m.fc_rh_mean_24h_obs - m.fc_rh_mean_24h_nbm)):.1f} %")
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(obs + 0.1, fc + 0.1, s=4, alpha=0.25, color=SERIES[0])
ax.plot([0.1, 100], [0.1, 100], color=MUTED, lw=1)
ax.set(xscale="log", yscale="log", xlabel="observed 24 h rain + 0.1 (mm)", ylabel="NBM forecast + 0.1 (mm)",
       title="NBM day-1 rain forecast vs observed (Apr to Dec 2025)");"""),
    md("""Temperature and humidity forecasts are close to reality; rain is the weak link. Since rain also drives most of the soil moisture error (notebook 02), forecast quality sets the ceiling for model skill (notebook 04)."""),
])

# =============================================================================================== 02
nb("02_relationships.ipynb", [
    md("""# 02 · Relationships: what actually drives 24 h soil moisture change?

Target: **Δ = soil moisture in 24 h − soil moisture now** at 5 cm, in percentage points (pp). Unfrozen soil only.
Each section ends with the feature it motivated (all in `soilcast/features.py`)."""),
    code(SETUP),
    code("""from soilcast.features import sm_valid, hourly_panel
from soilcast import data
d = pd.read_parquet(PROCESSED / "features_5cm_perfect.parquet")
v = d[sm_valid(d)].copy()
v["dy"] = 100 * (v.sm_t24 - v.sm)
v["sm_pct"] = 100 * v.sm
print(len(v), "issue times (every 3 h) across", v.station.nunique(), "stations")"""),
    md("## 1. The target is heavy-tailed: mostly tiny changes, occasional big jumps"),
    code("""print(v.dy.quantile([.01, .05, .25, .5, .75, .95, .99]).round(2).to_dict())
rain = v.fc_precip_24h >= 1
se = v.dy ** 2   # squared error of persistence (it predicts dy = 0)
print(f"|dy| < 0.5 pp: {(v.dy.abs() < .5).mean():.0%} of windows")
print(f"windows with >= 1 mm rain in next 24 h: {rain.mean():.0%}, share of persistence squared error: {se[rain].sum() / se.sum():.0%}")
fig, ax = plt.subplots()
ax.hist(v.dy.clip(-6, 20), bins=150, color=SERIES[0]); ax.set_yscale("log")
ax.set(xlabel="24 h change in soil moisture (pp, clipped to [-6, 20])", ylabel="count (log)", title="Distribution of the target");"""),
    md("""**So:** the problem is mostly "predict the rain response." A loss that downweights big jumps (Huber) is worth trying when tuning (notebook 07).

## 2. Dry-down: wetter soil dries faster (exponential decay)"""),
    code("""dry = v[(v.fc_precip_24h == 0) & (v.precip_24h == 0)].copy()
dry["texture"] = pd.cut(dry.sand, [0, 20, 50, 100], labels=["fine (sand < 20%)", "medium", "sandy (> 50%)"])
dry["sm_bin"] = pd.cut(dry.sm_pct, [0, 5, 10, 15, 20, 25, 30, 40, 60])
t = dry.groupby(["sm_bin", "texture"], observed=True).dy.mean().unstack()
fig, ax = plt.subplots()
for i, c in enumerate(t.columns):
    ax.plot([iv.mid for iv in t.index], t[c], marker="o", label=c, color=SERIES[i])
ax.axhline(0, color=MUTED, lw=.8)
ax.set(xlabel="soil moisture now (%)", ylabel="mean 24 h change (pp)", title="Dry-down rate vs wetness (no rain past or next 24 h)")
ax.legend(); t.round(2)"""),
    code("""dry["t_bin"] = pd.cut(dry.fc_air_t_mean_24h, [-10, 5, 10, 15, 20, 25, 35])
dry["wet"] = pd.cut(dry.sm_pct, [0, 15, 25, 60], labels=["dry (< 15%)", "moist (15-25%)", "wet (> 25%)"])
t2 = dry.groupby(["t_bin", "wet"], observed=True).dy.mean().unstack()
fig, ax = plt.subplots()
for i, c in enumerate(t2.columns):
    ax.plot([iv.mid for iv in t2.index], t2[c], marker="o", label=c, color=SERIES[i])
ax.set(xlabel="forecast mean air temp next 24 h (C)", ylabel="mean 24 h change (pp)", title="Evaporative demand speeds drying, if there is water to lose")
ax.legend();"""),
    md("""### Fitting the dry-down curve on individual events
For every rain event followed by 5 dry days (unfrozen), fit θ(t) − θ_floor = A·e^(−t/τ), with θ_floor fixed at the station's 2nd-percentile growing-season moisture (a stand-in for the driest the soil gets). Fixing the floor makes τ well-determined from 5 days of data; fitting the floor freely as well makes τ unstable (it trades off against the floor).
τ (hours) is the drying time constant: the excess water above the floor shrinks by ~63% every τ hours."""),
    code("""from scipy.stats import spearmanr
u = data.load_uscrn(); soils = data.load_soils()
h = hourly_panel(u, 5)
floor = h[h.st > 5].groupby("station").sm.quantile(0.02)
events = []
for st, g in h.groupby("station"):
    g = g.reset_index(drop=True)
    p24 = g.precip.rolling(24, min_periods=1).sum()
    nxt = g.precip[::-1].rolling(120, min_periods=100).sum()[::-1].shift(-1)
    starts = np.where((p24 >= 5) & (nxt < 0.3) & (g.st > 5))[0]
    last = -10**9
    for i in starts:
        if i - last < 120: continue
        seg = g.iloc[i:i + 120]
        if seg.sm.isna().mean() > .1 or (seg.st <= 1).any(): continue
        k = int(np.nanargmax(seg.sm.to_numpy()[:6])); seg = seg.iloc[k + 6:]   # skip ~6 h of fast drainage
        y = seg.sm.to_numpy() - floor[st]; t = np.arange(len(seg), dtype=float)
        ok = ~np.isnan(y) & (y > 0.002)
        if ok.sum() < 60 or y[ok][0] < 0.01: continue
        slope, icpt = np.polyfit(t[ok], np.log(y[ok]), 1)
        if slope >= 0: continue
        resid = y[ok] - np.exp(icpt + slope * t[ok])
        es = 0.6108 * np.exp(17.27 * seg.air_t / (seg.air_t + 237.3))
        events.append(dict(station=st, start=seg.time_utc.iloc[0], tau_h=-1 / slope, excess_pp=100 * y[ok][0],
                           fit_rmse_pp=100 * np.sqrt(np.mean(resid ** 2)), air_t=seg.air_t.mean(),
                           vpd_kpa=np.nanmean(es * (1 - seg.rh / 100))))
        last = i
ev = pd.DataFrame(events).merge(soils[soils.depth_cm == 5][["station", "sandtotal_r", "claytotal_r"]], on="station")
print(len(ev), "dry-down events; median fit RMSE", ev.fit_rmse_pp.median().round(3), "pp")
ev.groupby("station")[["tau_h", "excess_pp", "fit_rmse_pp", "sandtotal_r", "claytotal_r"]].median().sort_values("tau_h").round(1)"""),
    code("""fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
order = ev.groupby("station").tau_h.median().sort_values().index
a1.boxplot([ev[ev.station == s].tau_h for s in order], vert=False, showfliers=False)
a1.set_yticks(range(1, len(order) + 1), [s.split("_")[1] for s in order]); a1.set(xlabel="tau (h)", title="Drying time constant by station")
a2.scatter(ev.vpd_kpa, ev.tau_h, s=10, alpha=.5, color=SERIES[1]); a2.set_yscale("log")
rho, pv = spearmanr(ev.vpd_kpa, ev.tau_h, nan_policy="omit")
a2.set(xlabel="mean VPD during dry-down (kPa)", ylabel="tau (h, log)", title=f"tau vs evaporative demand (Spearman rho = {rho:.2f})")
print(f"Spearman(tau, VPD) = {rho:.2f} (p = {pv:.1g});  Spearman(tau, air temp) = {spearmanr(ev.air_t, ev.tau_h, nan_policy='omit')[0]:.2f}")"""),
    md("""Read the printed correlation before trusting any story here: a negative rho means higher demand → faster drying (smaller τ). Event-level τ is noisy (roots, crop stage and the floor approximation all vary), which is why the models use the binned relationships above rather than a fitted τ."""),
    md("""**Feature:** `sm_above_floor` (moisture above its 30-day minimum), `drying_demand` = that × forecast ET0 (Hargreaves, from forecast temps), `vpd`.

## 3. Rain response depends on soil texture and on how much room is left"""),
    code("""r = v[(v.fc_precip_24h >= 1) & (v.fc_precip_24h <= 30)]
slopes = r.groupby("station").apply(lambda g: pd.Series({"pp_per_mm": np.polyfit(g.fc_precip_24h, g.dy, 1)[0],
                                                         "clay": g.clay.iloc[0], "sand": g.sand.iloc[0], "n": len(g)}))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.scatter(slopes.clay, slopes.pp_per_mm, s=60, color=SERIES[0])
for st, s in slopes.iterrows():
    ax.annotate(st.split("_")[1], (s.clay, s.pp_per_mm), xytext=(4, 3), textcoords="offset points", fontsize=8)
ax.set(xlabel="clay % (SSURGO, 5 cm)", ylabel="moisture gain per mm of rain (pp/mm)", title="Rain response by station")
slopes.sort_values("pp_per_mm").round(2)"""),
    code("""big = v[v.fc_precip_24h >= 5].copy()
big["deficit"] = big.porosity - big.sm
big["deficit_bin"] = pd.qcut(big.deficit, 5)
big["rain_bin"] = pd.cut(big.fc_precip_24h, [5, 10, 20, 200])
t3 = big.groupby(["deficit_bin", "rain_bin"], observed=True).dy.mean().unstack()
fig, ax = plt.subplots()
for i, c in enumerate(t3.columns):
    ax.plot(range(len(t3)), t3[c], marker="o", label=f"rain {c} mm", color=SERIES[i])
ax.set_xticks(range(len(t3)), [f"{iv.left:.2f}-{iv.right:.2f}" for iv in t3.index])
ax.set(xlabel="storage deficit = SSURGO porosity - moisture (m3/m3)", ylabel="mean 24 h change (pp)",
       title="Same rain, bigger jump when the soil has more empty pore space"); ax.legend();"""),
    md("""**Feature:** `fc_rain_x_deficit`, `fc_rain_x_clay`, `fc_rain_x_sand`, `fc_rain_x_awc`, `fc_rain_log`. Tree models can find interactions alone, but with only 10 soils they learn them far better when handed explicitly.

## 4. A small daily cycle"""),
    code("""v["local_hour"] = (v.time_utc.dt.hour - 7) % 24   # Mountain Standard Time, approx for the region
dev = (100 * (v.sm - v.sm_24h_mean)).groupby(v.local_hour).mean()
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(dev.index, dev.values, marker="o", color=SERIES[0]); ax.axhline(0, color=MUTED, lw=.8)
ax.set(xlabel="local hour (approx.)", ylabel="pp vs 24 h mean", title="Diurnal cycle at 5 cm (part real, part sensor temperature sensitivity)");"""),
    md("""**Feature:** hour-of-day encodings (`hod_sin`, `hod_cos`). Only ±0.1 pp, much smaller than rain effects.

## 5. Which inputs move with the target? (rank correlation)"""),
    code("""from soilcast.features import FEATURE_SETS
cols = FEATURE_SETS["v2_engineered"]
corr = v[cols + ["dy"]].corr(method="spearman")["dy"].drop("dy").sort_values(key=abs, ascending=False)
fig, ax = plt.subplots(figsize=(7, 7))
top = corr.head(25)[::-1]
ax.barh(top.index, top.values, color=[SERIES[0] if x > 0 else SERIES[1] for x in top.values])
ax.set(xlabel="Spearman correlation with 24 h change", title="Top 25 features by |correlation|");"""),
    md("""Correlation only shows one input at a time; interactions such as rain × deficit matter more inside the models (see SHAP in notebook 03).

## 6. Deeper sensors change less"""),
    code("""rows = []
for dep in (5, 10, 20):
    x = pd.read_parquet(PROCESSED / f"features_{dep}cm_perfect.parquet", columns=["sm", "sm_t24", "st", "st_t24", "fc_precip_24h"])
    x = x[sm_valid(x)]; dd = 100 * (x.sm_t24 - x.sm)
    rows.append({"depth_cm": dep, "mean |24h change| pp": dd.abs().mean(), "std pp": dd.std(),
                 "mean change with >=5 mm rain": dd[x.fc_precip_24h >= 5].mean()})
pd.DataFrame(rows).round(2)"""),
    md("""## Summary
| Finding | Feature it motivated |
|---|---|
| Rain windows are ~20% of samples but most of the error | rain forecast features; Huber loss option |
| Dry-down is exponential, faster when wet and warm | `sm_above_floor`, `drying_demand`, `fc_et0_hargreaves`, `vpd` |
| Rain response varies ~3x across stations (lowest in pure sand, highest in heavy clay, but not a clean function of clay alone) and grows with empty pore space | `fc_rain_x_deficit`, `fc_rain_x_clay`, `fc_rain_x_sand`, `fc_rain_x_awc` |
| Small daily cycle | `hod_sin`, `hod_cos` |
| Deeper = smoother | train one model per depth |"""),
])

# =============================================================================================== 03
nb("03_baselines.ipynb", [
    md("""# 03 · Baselines: soil moisture 24 h ahead (her Model 1)

Models: **persistence** (no change), **ridge linear**, **LightGBM**. Learned models predict the change, then add the current value.
Inputs grow in four steps: sensors only → + forecast → + SSURGO soil → + engineered features (v2).

Forecast inputs here are **perfect** (observed weather): an upper bound. Notebook 04 swaps in real NBM forecasts.
Splits: train 2016 to 2023 → test 2024 to 2025 (all stations), and **new station** (leave one station out)."""),
    code(SETUP),
    code("""from soilcast.features import FEATURE_SETS, sm_valid
from soilcast.evaluate import time_split, loso_splits, sm_metrics, bootstrap_rmse_ci
from soilcast.models import fit_predict_delta, make_model
d = pd.read_parquet(PROCESSED / "features_5cm_perfect.parquet")
v = d[sm_valid(d)]
tr, te = time_split(v)
print(len(tr), "train rows,", len(te), "test rows")"""),
    md("## Time split: all feature sets × models"),
    code("""preds, models = {"persistence": te.sm.to_numpy()}, {}
for fs, cols in FEATURE_SETS.items():
    for kind in ("linear", "gbm"):
        preds[f"{kind} | {fs}"], models[(kind, fs)] = fit_predict_delta(kind, tr[cols], tr.sm, tr.sm_t24, te[cols], te.sm)
res = sm_metrics(te.sm_t24, preds, te.sm, te.fc_precip_24h)
res.to_csv(RESULTS / "03_time_split_5cm.csv", index=False)
res"""),
    code("""fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(res))
ax.bar(x - .2, res.RMSE_pp, .4, label="all test windows", color=[model_color(m) for m in res.model])
ax.bar(x + .2, res.RMSE_pp_rain_events, .4, label="rain >= 5 mm next 24 h", color=[model_color(m) for m in res.model], alpha=.45)
ax.set_xticks(x, [m.replace(" | ", "\\n") for m in res.model], fontsize=7.5)
ax.set(ylabel="RMSE (pp)", title="Test 2024-2025, 5 cm. Solid = all windows, faded = rain windows"); ax.legend();"""),
    md("""Read the table top to bottom: almost no gain from sensors alone, a big jump once the **forecast** is added, and further gains from soil + engineered features (especially for the linear model, which can't discover interactions alone).

### Is the difference real? Bootstrap CI by station-month"""),
    code("""groups = te.station + "_" + te.time_utc.dt.strftime("%Y-%m")
for name in ["persistence", "gbm | sensors+forecast+soil", "gbm | v2_engineered"]:
    lo, hi = bootstrap_rmse_ci(te.sm_t24, preds[name], groups, n=300)
    print(f"{name:32s} RMSE {100 * np.sqrt(np.mean((te.sm_t24 - preds[name]) ** 2)):.3f} pp   95% CI [{100 * lo:.3f}, {100 * hi:.3f}]")"""),
    md("## New station: train on 9 stations, test on the 10th (closest to deploying a new probe)"),
    code("""cols = FEATURE_SETS["v2_engineered"]
rows = []
for st, a, b in loso_splits(v):
    p = {"persistence": b.sm.to_numpy()}
    for kind in ("linear", "gbm"):
        p[kind], _ = fit_predict_delta(kind, a[cols], a.sm, a.sm_t24, b[cols], b.sm)
    r = sm_metrics(b.sm_t24, p, b.sm, b.fc_precip_24h); r.insert(0, "station", st); rows.append(r)
loso = pd.concat(rows)
loso.to_csv(RESULTS / "03_new_station_5cm.csv", index=False)
print(loso.groupby("model")[["RMSE_pp", "skill_vs_persistence", "RMSE_pp_rain_events"]].mean().round(3))
piv = loso.pivot(index="station", columns="model", values="skill_vs_persistence")
fig, ax = plt.subplots(figsize=(9, 4))
y = np.arange(len(piv))
ax.barh(y - .2, piv["linear"], .4, label="linear", color=model_color("linear"))
ax.barh(y + .2, piv["gbm"], .4, label="gbm", color=model_color("gbm"))
ax.set_yticks(y, piv.index, fontsize=8); ax.axvline(0, color=MUTED, lw=.8)
ax.set(xlabel="skill vs persistence (1 - RMSE/RMSE_persistence)", title="Held-out station skill, v2 features"); ax.legend();"""),
    md("## What the GBM learned: importance and SHAP"),
    code("""gbm = models[("gbm", "v2_engineered")]
imp = pd.Series(gbm.booster_.feature_importance("gain"), index=cols); imp = imp / imp.sum()
sample = te.sample(8000, random_state=0)
contrib = gbm.predict(sample[cols], pred_contrib=True)[:, :-1]   # SHAP values (LightGBM TreeSHAP)
shap = pd.DataFrame(contrib, columns=cols, index=sample.index)
mean_abs = (100 * shap.abs().mean()).sort_values()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
imp.sort_values().tail(15).plot.barh(ax=a1, color=SERIES[1]); a1.set(title="Gain importance (share)")
mean_abs.tail(15).plot.barh(ax=a2, color=SERIES[0]); a2.set(title="Mean |SHAP| (pp of predicted change)")
plt.tight_layout()"""),
    code("""fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
sc = a1.scatter(sample.fc_precip_24h, 100 * shap.fc_rain_x_deficit, c=sample.porosity - sample.sm, s=6, cmap="viridis")
plt.colorbar(sc, ax=a1, label="storage deficit (m3/m3)")
a1.set(xscale="symlog", xlabel="forecast rain next 24 h (mm)", ylabel="SHAP of rain x deficit (pp)", title="Rain effect grows with empty pore space")
sc2 = a2.scatter(100 * sample.sm_above_floor, 100 * shap.drying_demand, c=sample.fc_et0_hargreaves, s=6, cmap="magma_r")
plt.colorbar(sc2, ax=a2, label="forecast ET0 (mm/day)")
a2.set(xlabel="moisture above 30-day floor (pp)", ylabel="SHAP of drying demand (pp)", title="Dry-down learned as wetness x demand");"""),
    md("## Where does the model still miss?"),
    code("""best = preds["gbm | v2_engineered"]
e = te.assign(err=100 * (best - te.sm_t24), perr=100 * (te.sm - te.sm_t24))
e["rain_bin"] = pd.cut(e.fc_precip_24h, [-0.1, 0.1, 1, 5, 15, 300], labels=["0", "0-1", "1-5", "5-15", "15+"])
t = e.groupby("rain_bin", observed=True).agg(n=("err", "size"), gbm_rmse=("err", lambda s: np.sqrt((s ** 2).mean())),
                                             persistence_rmse=("perr", lambda s: np.sqrt((s ** 2).mean())), gbm_bias=("err", "mean"))
t.round(3)"""),
    md("""Large rain events remain the hardest: the model under-predicts the biggest jumps (negative bias). Huber loss and per-texture tuning are levers in notebook 07.

## Example: one summer, predicted vs actual"""),
    code("""ex = te.assign(pred=best)
ex = ex[(ex.station == "SD_Buffalo_13_ESE") & (ex.time_utc >= "2025-05-01") & (ex.time_utc < "2025-09-01")]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(ex.time_utc + pd.Timedelta("24h"), 100 * ex.sm_t24, color="black", lw=1.2, label="actual")
ax.plot(ex.time_utc + pd.Timedelta("24h"), 100 * ex.sm, color=model_color("persistence"), lw=1, label="persistence")
ax.plot(ex.time_utc + pd.Timedelta("24h"), 100 * ex.pred, color=model_color("gbm"), lw=1.2, label="GBM v2 (issued 24 h earlier)")
ax.set(ylabel="soil moisture 5 cm (%)", title="SD Buffalo, summer 2025 (perfect forecast inputs)"); ax.legend();"""),
    md("## Other depths (GBM v2 vs persistence)"),
    code("""rows = []
for dep in (10, 20):
    x = pd.read_parquet(PROCESSED / f"features_{dep}cm_perfect.parquet"); x = x[sm_valid(x)]
    a, b = time_split(x)
    p, _ = fit_predict_delta("gbm", a[cols], a.sm, a.sm_t24, b[cols], b.sm)
    r = sm_metrics(b.sm_t24, {"persistence": b.sm, "gbm | v2_engineered": p}, b.sm, b.fc_precip_24h); r.insert(0, "depth_cm", dep); rows.append(r)
pd.concat(rows)"""),
    code("""# Save test predictions for comparison with the LSTM (notebook 08)
out = te[["station", "time_utc", "sm", "sm_t24", "fc_precip_24h"]].assign(
    pred_gbm_v2=preds["gbm | v2_engineered"], pred_linear_v2=preds["linear | v2_engineered"])
out.to_parquet(RESULTS / "03_test_predictions_5cm.parquet")"""),
])

# =============================================================================================== 04
nb("04_realistic_forecasts.ipynb", [
    md("""# 04 · Realistic forecasts: how much skill survives real NBM forecasts?

Everything so far fed the model **observed** future weather. Here the same models (trained 2016 to 2023) get **archived NOAA National Blend of Models (NBM)** forecasts instead:
- 24 h window: forecast valid at each hour, issued ~24 h earlier (lead day 1)
- 72 h window: lead day 1 / 2 / 3 for hours 1-24 / 25-48 / 49-72

Test period: **Apr to Dec 2025** (unfrozen soil). The NBM archive starts 2024-10-09, so Oct 2024 to Mar 2025 is used only to estimate forecast error for the augmentation experiment."""),
    code(SETUP),
    code("""from soilcast.features import FEATURE_SETS, sm_valid
from soilcast.evaluate import sm_metrics
from soilcast.models import fit_predict_delta, make_model
P = pd.read_parquet(PROCESSED / "features_5cm_perfect.parquet")
N = pd.read_parquet(PROCESSED / "features_5cm_nbm.parquet")
train = P[sm_valid(P) & P.year.between(2016, 2023)]
key = ["station", "time_utc"]
test_p = P[sm_valid(P) & (P.time_utc >= "2025-04-01")]
test_n = N[sm_valid(N) & (N.time_utc >= "2025-04-01")]
test_p = test_p.merge(test_n[key], on=key); test_n = test_n.merge(test_p[key], on=key)   # identical rows
test_p, test_n = test_p.sort_values(key).reset_index(drop=True), test_n.sort_values(key).reset_index(drop=True)
assert (test_p.sm.values == test_n.sm.values).all()
print(len(train), "train rows;", len(test_p), "test rows (same rows in both test sets)")"""),
    code("""rows = []
for fs in ["sensors_24h", "sensors+forecast", "sensors+forecast+soil", "v2_engineered"]:
    cols = FEATURE_SETS[fs]
    for kind in ("linear", "gbm"):
        m = make_model(kind).fit(train[cols], train.sm_t24 - train.sm)
        for src, t in (("perfect", test_p), ("NBM", test_n)):
            r = sm_metrics(t.sm_t24, {"persistence": t.sm, "model": t.sm + m.predict(t[cols])}, t.sm,
                           test_p.fc_precip_24h)  # rain-event subset defined by OBSERVED rain for both
            r = r.iloc[1:].assign(model=f"{kind} | {fs}", forecast=src)
            rows.append(r)
pers = sm_metrics(test_p.sm_t24, {"persistence": test_p.sm}, test_p.sm, test_p.fc_precip_24h).assign(forecast="-")
res = pd.concat([pers] + rows, ignore_index=True)
res.to_csv(RESULTS / "04_perfect_vs_nbm_5cm.csv", index=False)
res[["model", "forecast", "RMSE_pp", "skill_vs_persistence", "RMSE_pp_rain_events", "n"]]"""),
    code("""piv = res[res.forecast != "-"].pivot(index="model", columns="forecast", values="skill_vs_persistence")
piv = piv.loc[[f"{k} | {fs}" for fs in ["sensors_24h", "sensors+forecast", "sensors+forecast+soil", "v2_engineered"] for k in ("linear", "gbm")]]
fig, ax = plt.subplots(figsize=(9, 4))
y = np.arange(len(piv))
ax.barh(y + .2, piv["perfect"], .4, color=SERIES[0], label="perfect (observed weather)")
ax.barh(y - .2, piv["NBM"], .4, color=SERIES[1], label="real NBM forecast")
ax.set_yticks(y, piv.index, fontsize=8); ax.axvline(0, color=MUTED, lw=.8)
ax.set(xlabel="skill vs persistence", title="Apr-Dec 2025, 5 cm: skill with perfect vs real forecasts"); ax.legend();"""),
    md("""## Can training on noisy forecasts help?
We have no archived forecasts for 2016 to 2023, so the model learned to trust rain inputs completely. One fix: **add realistic forecast error to the training inputs** so the model learns to hedge. Error statistics come from NBM vs observed for Oct 2024 to Mar 2025 only (no overlap with the test)."""),
    code("""cal = P.merge(N, on=key, suffixes=("_o", "_f"))
cal = cal[(cal.time_utc < "2025-04-01")][["fc_precip_24h_o", "fc_precip_24h_f"]].dropna()
o, f = cal.fc_precip_24h_o, cal.fc_precip_24h_f
wet = o >= 1
p_miss = ((f < 0.5) & wet).sum() / wet.sum()                 # forecast ~dry when it rained
p_false = ((f >= 1) & (o < 0.5)).sum() / (o < 0.5).sum()      # forecast rain when dry
ratio = np.log((f[wet & (f >= 0.5)] + .1) / (o[wet & (f >= 0.5)] + .1))
print(f"calibration windows {len(cal)}; P(miss | rain) {p_miss:.2f}; P(false alarm | dry) {p_false:.3f}; "
      f"log-ratio mean {ratio.mean():.2f} sd {ratio.std():.2f}")
false_amounts = f[(f >= 1) & (o < 0.5)].to_numpy()

def corrupt(df, rng):
    df = df.copy()
    r = df.fc_precip_24h.to_numpy().copy()
    wet = r >= 1
    miss = wet & (rng.random(len(r)) < p_miss)
    noisy = r * np.exp(rng.normal(ratio.mean(), ratio.std(), len(r)))
    r_new = np.where(wet, np.where(miss, 0.0, noisy), r)
    fa = (r < 0.5) & (rng.random(len(r)) < p_false)
    r_new[fa] = rng.choice(false_amounts, fa.sum())
    scale = np.where(r > 0, r_new / np.maximum(r, 1e-6), 0)
    df["fc_precip_6h"] = np.where(fa, r_new / 4, df.fc_precip_6h * scale)
    df["fc_precip_24h"] = r_new
    rr = np.clip(r_new, None, 50)
    for c, base in (("fc_rain_x_deficit", df.porosity - df.sm), ("fc_rain_x_clay", df.clay),
                    ("fc_rain_x_awc", df.awc), ("fc_rain_x_sand", df.sand)):
        df[c] = rr * base
    df["fc_rain_log"] = np.log1p(rr)
    return df

rng = np.random.default_rng(7)
cols = FEATURE_SETS["v2_engineered"]
aug = pd.concat([train, corrupt(train, rng)], ignore_index=True)    # clean + corrupted copy
m_aug = make_model("gbm").fit(aug[cols], aug.sm_t24 - aug.sm)
m_clean = make_model("gbm").fit(train[cols], train.sm_t24 - train.sm)
out = sm_metrics(test_n.sm_t24, {"persistence": test_n.sm,
                                 "gbm v2, trained clean": test_n.sm + m_clean.predict(test_n[cols]),
                                 "gbm v2, trained with forecast noise": test_n.sm + m_aug.predict(test_n[cols])},
                 test_n.sm, test_p.fc_precip_24h)
out.to_csv(RESULTS / "04_noise_augmentation_5cm.csv", index=False)
out"""),
    md("""## Takeaways
- The realistic numbers (NBM rows) are the ones to report. The perfect-forecast numbers are the ceiling a better rain forecast could reach.
- Rain forecast skill is the bottleneck: NBM 24 h rain correlates ~0.6 with observed (notebook 01).
- If noise augmentation helps, train with it by default. Even better: archive the NWS forecasts for Jolly Lane from now on (`scripts/archive_nws_forecast.py`) so future models train on real forecasts."""),
])

# =============================================================================================== 05
nb("05_nitrate_loss_index.ipynb", [
    md("""# 05 · Nitrate loss risk index, next 72 h (proposed replacement for the NPK model)

**What it is:** a heuristic built from published numbers, not measured nitrogen loss.
- Wetness factor: 0 below 60% water-filled pore space (WFPS), 1 at ≥ 80%, linear between (denitrification turns on sharply at 60 to 80% WFPS; HESS 8:686, 2004)
- Temperature rate: % of soil nitrate lost per saturated day; Purdue Extension: < 55 °F 1-2, 55-65 °F 2-3, > 65 °F 4-5 (midpoints used)
- Index = sum over the next 72 h of hourly (wetness × rate / 24) → **% of soil nitrate at risk in the next 3 days**

WFPS = moisture / porosity, porosity = 1 − bulk density / 2.65 (SSURGO). Computed at 10 cm (where fertilizer N sits)."""),
    code(SETUP),
    code("""from soilcast.targets import nitrate_loss_rate, WFPS_ON, WFPS_FULL
wf = np.linspace(0.3, 1.0, 141); tc = np.linspace(0, 30, 121)
W, T = np.meshgrid(wf, tc)
R = nitrate_loss_rate(W * 0.5, T, np.full_like(W, 0.5))
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.pcolormesh(wf * 100, tc, R, cmap="Oranges", shading="auto"); plt.colorbar(im, label="% nitrate per day")
ax.set(xlabel="water-filled pore space (%)", ylabel="soil temperature (C)", title="Daily loss rate used in the index"); ax.grid(False);"""),
    code("""from soilcast.features import SENSORS, SOIL, FC72, NLRI_EXTRA, nlri_valid
from soilcast.evaluate import time_split, loso_splits, nlri_metrics
from soilcast.models import fit_predict_level
d = pd.read_parquet(PROCESSED / "features_10cm_perfect.parquet")
v = d[nlri_valid(d)]
print(f"windows with risk > 0.05%: {(v.nlri_72h > .05).mean():.1%};  90th pct {v.nlri_72h.quantile(.9):.2f}%;  max {v.nlri_72h.max():.1f}%")
m = v.assign(month=v.time_utc.dt.month).groupby(["station", "month"]).nlri_72h.apply(lambda s: (s > .05).mean()).unstack()
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(m.values, aspect="auto", cmap="Oranges", vmin=0); plt.colorbar(im, label="share of windows with risk")
ax.set_yticks(range(len(m)), m.index, fontsize=8); ax.set_xticks(range(m.shape[1]), m.columns)
ax.set(xlabel="month", title="When and where nitrate loss risk occurs (10 cm)"); ax.grid(False);"""),
    md("""Risk concentrates in spring (wet soils) and at fine-textured, wetter stations; the sandy Nebraska Sandhills station almost never reaches the WFPS threshold.

## Models (perfect forecasts, time split and new station)"""),
    code("""base = SENSORS + SOIL + NLRI_EXTRA
fsets = {"sensors+soil": base, "sensors+soil+forecast72h": base + FC72}
tr, te = time_split(v)
preds = {"persistence": 72 * te.nl_rate_h.to_numpy()}
for fs, cols in fsets.items():
    for kind in ("linear", "gbm"):
        preds[f"{kind} | {fs}"], _ = fit_predict_level(kind, tr[cols], tr.nlri_72h, te[cols])
res = nlri_metrics(te.nlri_72h, preds, te.nl_past24)
res.to_csv(RESULTS / "05_nlri_time_split.csv", index=False); res"""),
    md("""`AUC_new_risk` is the important column: among windows with **no risk in the past 24 h**, can the model tell which ones are about to become risky? Persistence can't (0.5)."""),
    code("""cols = fsets["sensors+soil+forecast72h"]
rows = []
for st, a, b in loso_splits(v):
    if (b.nlri_72h > .05).sum() < 20: continue
    p = {"persistence": 72 * b.nl_rate_h.to_numpy()}
    p["gbm"], _ = fit_predict_level("gbm", a[cols], a.nlri_72h, b[cols])
    r = nlri_metrics(b.nlri_72h, p, b.nl_past24); r.insert(0, "station", st); rows.append(r)
loso = pd.concat(rows); loso.to_csv(RESULTS / "05_nlri_new_station.csv", index=False)
loso.pivot(index="station", columns="model", values=["AUC_any_risk", "AUC_new_risk"]).round(3)"""),
    md("## With real NBM forecasts (Apr to Dec 2025)"),
    code("""N = pd.read_parquet(PROCESSED / "features_10cm_nbm.parquet")
key = ["station", "time_utc"]
tp = v[v.time_utc >= "2025-04-01"]
tn = N[nlri_valid(N) & (N.time_utc >= "2025-04-01")]
tp = tp.merge(tn[key], on=key).sort_values(key).reset_index(drop=True); tn = tn.merge(tp[key], on=key).sort_values(key).reset_index(drop=True)
m = __import__("soilcast.models", fromlist=["make_model"]).make_model("gbm").fit(tr[cols], tr.nlri_72h)
real = nlri_metrics(tp.nlri_72h, {"persistence": 72 * tp.nl_rate_h, "gbm, perfect forecast": np.clip(m.predict(tp[cols]), 0, None),
                                  "gbm, NBM forecast": np.clip(m.predict(tn[cols]), 0, None)}, tp.nl_past24)
real.to_csv(RESULTS / "05_nlri_nbm_2025.csv", index=False); real"""),
    code("""st_pick = (tp.assign(r=tp.nlri_72h > .05).groupby("station").r.sum().idxmax())
ex = tp.assign(perfect=np.clip(m.predict(tp[cols]), 0, None), nbm=np.clip(m.predict(tn[cols]), 0, None))
ex = ex[ex.station == st_pick]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(ex.time_utc, ex.nlri_72h, color="black", lw=1.2, label="actual index (next 72 h)")
ax.plot(ex.time_utc, ex.perfect, color=SERIES[0], lw=1, label="GBM, perfect forecast")
ax.plot(ex.time_utc, ex.nbm, color=SERIES[1], lw=1, label="GBM, NBM forecast")
ax.set(ylabel="% of soil nitrate at risk, next 72 h", title=f"{st_pick}, 2025"); ax.legend();"""),
    md("""## Caveats
- The index is computed from moisture and temperature, so predicting it is predicting when those cross published thresholds. It's a decision index, not validated N loss. Validation needs lab nitrate samples or N₂O flux measurements.
- SSURGO porosity doesn't always match the actual site (some readings exceed mapped porosity). A bulk density sample at Jolly Lane fixes this for the project site."""),
])

# =============================================================================================== 06
nb("06_npk_ec_check.ipynb", [
    md("""# 06 · Can her NPK model work? An EC check on real farm data

Low-cost NPK probes (DFRobot SEN0605 / JXCT class) convert **one conductivity signal** into N, P and K at fixed ratios (Baldi et al. 2026, *Horticulturae*). So "forecast the NPK readings" = "forecast EC."

No public continuous NPK-probe dataset was available, so we use the **Cook Agronomy Farm** (WSU): 42 field sites, daily bulk EC, moisture and soil temp at 0.3 m (Decagon 5TE), 2011 to 2012."""),
    code(SETUP),
    code("""from soilcast.cookfarm import daily_panel, EC_FEATURES
from soilcast.models import fit_predict_delta
from soilcast.evaluate import rmse
p = daily_panel(port=1)
x = p.dropna(subset=["ec_uScm", "vw", "st"])
print(len(x), "site-days with EC, moisture and temperature;", x.SOURCEID.nunique(), "sites")"""),
    md("## How much of EC is just moisture and temperature?"),
    code("""from sklearn.linear_model import LinearRegression
r2 = {}
for sid, g in x.groupby("SOURCEID"):
    if len(g) > 100:
        r2[sid] = LinearRegression().fit(g[["vw", "st"]], g.ec_uScm).score(g[["vw", "st"]], g.ec_uScm)
r2 = pd.Series(r2)
print(f"within-site R2 of EC ~ moisture + temperature: median {r2.median():.2f}, IQR [{r2.quantile(.25):.2f}, {r2.quantile(.75):.2f}]")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
for ax, sid in zip(axes, r2.sort_values().index[[5, len(r2) // 2, -5]]):
    g = x[x.SOURCEID == sid]
    sc = ax.scatter(100 * g.vw, g.ec_uScm, c=g.st, s=8, cmap="coolwarm")
    ax.set(title=f"{sid}  (R2 {r2[sid]:.2f})", xlabel="moisture (%)", ylabel="bulk EC (uS/cm)")
plt.colorbar(sc, ax=axes, label="soil temp (C)");"""),
    md("""Sensor resolution matters too: the 5TE reports EC in 0.01 dS/m steps (= 10 µS/cm), so small day-to-day changes are quantization.

## Forecasting EC 1, 2, 3 days ahead"""),
    code("""rows = []
for hzn in (1, 2, 3):
    cols = EC_FEATURES + [f"fc_precip_{hzn}d", f"fc_tmax_{hzn}d"]
    q = p.dropna(subset=["ec_uScm", f"ec_t{hzn}"])
    tr, te = q[q.year == 2011], q[q.year == 2012]
    pr = {"persistence": te.ec_uScm.to_numpy()}
    for kind in ("linear", "gbm"):
        pr[kind], _ = fit_predict_delta(kind, tr[cols], tr.ec_uScm, tr[f"ec_t{hzn}"], te[cols], te.ec_uScm)
    base = rmse(te[f"ec_t{hzn}"], pr["persistence"])
    for k, yhat in pr.items():
        rows.append({"horizon_days": hzn, "model": k, "RMSE_uScm": rmse(te[f"ec_t{hzn}"], yhat),
                     "MAE_uScm": np.mean(np.abs(te[f"ec_t{hzn}"] - yhat)), "skill_vs_persistence": 1 - rmse(te[f"ec_t{hzn}"], yhat) / base})
ec = pd.DataFrame(rows); ec.to_csv(RESULTS / "06_ec_forecast.csv", index=False); ec.round(3)"""),
    md("""## Takeaways
- Within a site, moisture and temperature explain most of EC's day-to-day variation, so EC adds little independent nutrient information.
- EC forecasts barely beat persistence at any horizon. A model predicting the probe's N, P and K would inherit this, times three fixed ratios.
- **Recommendation:** replace the NPK forecast with the nitrate loss risk index (notebook 05), and use NPK/EC readings only as a "pull a soil test here" flag until lab samples exist to validate them."""),
])

# =============================================================================================== 07
nb("07_tuning.ipynb", [
    md("""# 07 · Hyperparameter tuning (Optuna)

Protocol (no test leakage):
1. **Expanding-window CV inside the training years:** fit on years < Y, validate on Y, for Y = 2020, 2021, 2022, 2023. The objective is the mean validation RMSE.
2. Tree-structured Parzen search with median pruning (bad trials stop after 1-2 folds).
3. Refit the best params on 2016 to 2023 and score **once** on 2024 to 2025.

The search space includes **Huber loss** (with a threshold on the right scale) because the target is heavy-tailed (notebook 02).

This notebook runs a short study so it finishes on a laptop CPU. For the real search, use the script (resumable, GPU-capable):
```
python scripts/tune_gbm.py --trials 300 --storage sqlite:///results/tuning.db                 # LightGBM, CPU
python scripts/tune_gbm.py --kind xgb --device cuda --trials 500 --storage sqlite:///results/tuning.db   # XGBoost, GPU
python scripts/tune_gbm.py --target nlri --depth 10 --trials 200                               # nitrate index
```"""),
    code(SETUP),
    code("""import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from soilcast.features import FEATURE_SETS, sm_valid
from soilcast.tuning import run_study, params_from_trial
from soilcast.evaluate import time_split, sm_metrics
from soilcast.models import make_model
N_TRIALS = 15          # raise to 200+ on Colab / RTX
MAX_ROWS = 60_000      # subsample per fold for speed; set None for full data
KIND, DEVICE = "gbm", "cpu"   # KIND="xgb", DEVICE="cuda" on a GPU
d = pd.read_parquet(PROCESSED / "features_5cm_perfect.parquet"); v = d[sm_valid(d)]
cols = FEATURE_SETS["v2_engineered"]"""),
    code("""study = run_study(v, cols, "sm_t24", "sm", kind=KIND, n_trials=N_TRIALS, device=DEVICE, max_rows=MAX_ROWS)
print("best CV RMSE (pp):", round(100 * study.best_value, 4))
best = params_from_trial(study.best_params, KIND); best"""),
    code("""from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
plot_optimization_history(study); plt.title("CV RMSE by trial")
if len([t for t in study.trials if t.state.name == "COMPLETE"]) > 3:
    plot_param_importances(study)
study.trials_dataframe()[["number", "value", "state", "params_objective", "params_num_leaves", "params_learning_rate", "params_n_estimators"]].sort_values("value").head(10)"""),
    md("## Refit on all training years, score once on the test years"),
    code("""tr, te = time_split(v)
default = make_model(KIND, device=DEVICE).fit(tr[cols], tr.sm_t24 - tr.sm)
tuned = make_model(KIND, best, DEVICE).fit(tr[cols], tr.sm_t24 - tr.sm)
res = sm_metrics(te.sm_t24, {"persistence": te.sm, "gbm default": te.sm + default.predict(te[cols]),
                             "tuned " + KIND: te.sm + tuned.predict(te[cols])}, te.sm, te.fc_precip_24h)
res.to_csv(RESULTS / "07_tuned_vs_default_5cm.csv", index=False); res"""),
    md("""## What else to tune (ideas, in rough order of expected payoff)
- **Loss:** Huber threshold, or a two-part model (P(rain response) × size), since rain windows dominate error
- **Feature windows:** past window length (24 h vs 72 h), rain thresholds, 30-day floor window
- **Per-texture models** or texture-stratified sample weights
- **Training with forecast noise** (notebook 04) at different noise levels, tuned on NBM-period validation
- **Multi-depth model:** one model with depth as a feature, trained on 5/10/20 cm jointly"""),
])

# =============================================================================================== 08
nb("08_lstm.ipynb", [
    md("""# 08 · LSTM sequence model

**Why try it:** an LSTM reads the raw hourly history (72 h) and the hourly forecast (next 24 h) directly, so it can learn timing (e.g. rain at hour 2 vs hour 22) and shapes the tabular features summarize away. Most soil moisture forecasting papers report an LSTM, so reviewers will expect one.

**Why it may not win:** we have 10 stations and engineered features already encode the physics; gradient boosting is often as good on tabular data of this size. Either outcome is a useful result.

Architecture (`soilcast/seq.py`):
```
past 72 h  [sm, st, air_t, rh, log rain, dsm_1h, missing flag] -> LSTM (2 layers) --+
next 24 h  [log rain, air_t, rh] forecast                     -> LSTM (1 layer)  --+-> MLP -> change at +1..+24 h
SSURGO     [sand, silt, clay, awc, drainage, bulk density]    -> Linear+ReLU     --+
```
Loss: Huber on all 24 horizons (masked where missing); early stopping on 2023; test 2024 to 2025.

**Run modes:** `SMOKE` trains a small model on a subsample (CPU, a few minutes) to prove the pipeline. Set `MODE = "FULL"` on Colab (A100) or the RTX 4080 Super, or run `python scripts/train_lstm.py --device cuda` / `--sweep 20`."""),
    code(SETUP + """
# Colab: Runtime > Change runtime type > GPU. torch is preinstalled on Colab.
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")"""),
    code("""import os
from soilcast import data
from soilcast.seq import hourly_for_seq, train_lstm
from soilcast.evaluate import sm_metrics
MODE = "FULL" if torch.cuda.is_available() else "SMOKE"
CFG = {"SMOKE": dict(past_len=48, hidden=64, fut_hidden=32, layers=1, epochs=4, batch_size=512, max_train=60_000,
                     train_step_h=3, num_workers=0, patience=3),
       "FULL":  dict(past_len=72, hidden=128, fut_hidden=64, layers=2, epochs=30, batch_size=1024, max_train=None,
                     train_step_h=1, num_workers=0 if os.name == "nt" else 4, patience=5)}[MODE]
print(MODE, CFG)
u, soils, nbm = data.load_uscrn(), data.load_soils(), data.load_forecasts()
h = hourly_for_seq(u, soils, 5)
h_nbm = hourly_for_seq(u[u.time_utc >= "2024-09-01"], soils, 5, forecasts=nbm)"""),
    code("""model, hist, preds, n_train = train_lstm(h, {"perfect": h, "nbm": h_nbm}, **CFG)
print("trained on", n_train, "windows")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(hist.epoch, hist.valid_rmse_pp_24h, marker="o", color=SERIES[2]); ax.set(xlabel="epoch", ylabel="valid RMSE at +24 h (pp)", title="Validation (2023)");"""),
    md("## Compare with persistence and GBM on the same test windows"),
    code("""g = pd.read_parquet(RESULTS / "03_test_predictions_5cm.parquet")
m = preds["perfect"].merge(g, on=["station", "time_utc"], suffixes=("_lstm", ""))
cmp = sm_metrics(m.sm_t24, {"persistence": m.sm, "gbm v2 (tabular)": m.pred_gbm_v2, "linear v2": m.pred_linear_v2,
                            f"lstm ({MODE})": m.pred_sm_t24}, m.sm, m.fc_precip_24h)
cmp.to_csv(RESULTS / f"08_lstm_vs_gbm_{MODE.lower()}.csv", index=False)
print(len(m), "shared test windows (perfect forecasts)"); cmp"""),
    code("""# Real NBM forecasts, Apr-Dec 2025 (LSTM only; see notebook 04 for the tabular models on NBM)
pn = preds["nbm"]; pn = pn[pn.time_utc >= "2025-04-01"]
tgt = h[["station", "time_utc", "sm"]].assign(sm_t24=h.groupby("station").sm.shift(-24),
       rain_next24=h.groupby("station").precip.transform(lambda s: s[::-1].rolling(24, min_periods=12).sum()[::-1].shift(-1)))
mn = pn.merge(tgt.drop(columns="sm"), on=["station", "time_utc"]).dropna(subset=["sm_t24"])
sm_metrics(mn.sm_t24, {"persistence": mn.sm, f"lstm ({MODE}), NBM": mn.pred_sm_t24}, mn.sm, mn.rain_next24)"""),
    md("""## Notes for the full run
- In `SMOKE` mode the LSTM sees ~60k windows for a few epochs: its numbers are a pipeline check, **not** a fair comparison. Judge the LSTM on the `FULL` run.
- Worth sweeping: past window (24/48/72/168 h), hidden size, layers, dropout, learning rate (`scripts/train_lstm.py --sweep 20`).
- Other sequence options if the LSTM is promising: a Temporal Fusion Transformer, or zero-shot time-series foundation models (TimesFM, Chronos) as a no-training baseline; a NeurIPS 2026 workshop paper found TimesFM strong below 1 day and weak beyond it on soil moisture.
- A fair final comparison uses the same test windows, the same forecast source (perfect and NBM), and bootstrap CIs (notebook 03)."""),
])
