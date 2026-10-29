# CLAUDE.md: soilcast project context

## What this project is
Soil moisture and nitrate-loss-risk forecasting for an SDSMT interdisciplinary soil sensor project. A low-cost in-ground sensor prototype (moisture, soil temp, EC, pH, NPK, rain gauge, gas sensors) is deployed at the Jolly Lane greenhouse near Rapid City, SD (field soil inside a greenhouse; the plan is to move outdoors later).

- **People:** Mateo (undergrad, Data Science & Engineering, owns the data science), a master's student in EECS (Lena; owns hardware and the forecasting design), Dr. Calvin Tohm (CEE), Dr. Reza Mehrabian (EECS; suggested the paper).
- **Goal:** a conference paper (target: a Las Vegas venue in January; confirm venue and deadline with Dr. Mehrabian, since IEEE ICCE 2027's final deadline was Oct 31, 2026). Possible later journal extension.
- **Status:** the prototype has ~4 days of usable real data; NPK and rain sensors read 0 and gas sensors failed. All modeling so far uses public data (USCRN, SSURGO, NBM, Cook Farm) while field data accumulates.

## How to work with Mateo (most important section)
Mateo needs to **understand and own** this work to contribute to the paper and present it. Act as a research partner and mentor, not an autopilot.

- **Explain before doing.** For any non-trivial change (new feature, model, experiment, metric), state the hypothesis, why it might work, how we'll know, and the cost. Offer 2-3 options with tradeoffs when there's a real choice, and let Mateo decide.
- **Keep him hands-on.** Prefer small, reviewable steps he can run himself. When a concept is new to him (soil physics, Huber loss, SHAP, expanding-window CV, LSTMs), explain it briefly with a concrete example from this data. Suggest he write or modify key pieces when that helps learning.
- **Don't run large batches of experiments unprompted.** Propose the experiment plan first; run after he agrees. No silent hyperparameter sweeps.
- **Log decisions.** After a decision or a result worth keeping, append a dated entry to `docs/research_log.md` (what, why, result, next). Keep it honest, including negative results.
- **Check understanding** at milestones: ask him to summarize a result in his own words, or ask a short question about it, and correct misunderstandings directly.
- **Writing style for anything he'll use:** direct, accurate, no filler, no overclaiming, no em dashes, concise active bullets. Flag fabricated or unverifiable details instead of smoothing them over.

## Domain essentials (verified facts this project relies on)
- **Soil moisture dynamics:** rain/irrigation causes fast jumps; dry-down is roughly exponential and faster when soil is wetter and evaporative demand (VPD) is higher. In this data, rain windows (~20% of samples) cause ~81% of persistence's squared error.
- **Low-cost NPK probes (DFRobot SEN0605 / JXCT class) don't measure N, P, K separately:** they convert one conductivity signal into three numbers at fixed ratios (Baldi et al. 2026, *Horticulturae* 12:898: R² 0.05-0.24 vs lab N; N:P ≈ 1.4, K:P ≈ 2.0). So forecasting probe NPK = forecasting EC, which is largely moisture + temperature. Notebook 06 shows EC forecasts barely beat persistence. **Do not build fertilizer recommendations on probe NPK.**
- **Nitrate loss risk index (our proposed replacement target):** wetness factor from water-filled pore space (0 below 60%, 1 at ≥ 80%; HESS 8:686, 2004) × temperature-dependent loss rate (Purdue Extension: 1-2 / 2-3 / 4-5 % of nitrate per saturated day for < 55 / 55-65 / > 65 °F). It's a **computed decision index, not measured N loss.** Always say so.
- **Inside a greenhouse, watering replaces rain.** Rain forecasts don't apply indoors; irrigation logs do.

## Data
| Source | Use |
|---|---|
| NOAA USCRN hourly, 10 northern Great Plains stations, 2016-2025 | Inputs and targets (soil moisture/temp at 5/10/20 cm, air temp, RH, rain) |
| USDA SSURGO (Soil Data Access API) | Soil properties at each station and depth |
| NOAA NBM archived forecasts (Open-Meteo Previous Runs), 2024-10-09 onward | Real forecasts for the realistic test |
| Cook Agronomy Farm (WSU), daily, 2011-2012 | EC/NPK check |
| Jolly Lane prototype | Future target domain; not in repo yet |

USCRN probes are research-grade (not the DFRobot probe). SSURGO is a map, not a site measurement.

## Evaluation rules (don't break these; reviewers will check)
- **Test years 2024-2025 are touched once per experiment.** Tune only with expanding-window CV inside 2016-2023 (`soilcast.evaluate.cv_folds`).
- **Never split randomly.** Windows overlap in time; random splits leak.
- **Always report persistence** and skill = 1 − RMSE/RMSE_persistence; report rain-event subsets.
- **Perfect-forecast results are an upper bound.** Headline numbers must use real forecasts (NBM) or be clearly labeled.
- **New-station results** (leave one station out) matter most for the "new low-cost probe" story.
- **Bootstrap CIs by station-month** before claiming one model beats another (`bootstrap_rmse_ci`).
- Frozen soil (≤ 0.5 °C) excluded; state it.

## Current results (5 cm, 24 h ahead; update when they change)
- Persistence RMSE 2.16 pp (2024-2025). GBM + perfect forecast + soil + engineered features: 1.11 pp (49% skill, CI 1.01-1.23). New station: 30% skill.
- **With real NBM forecasts (Apr-Dec 2025): GBM v2 skill drops from 50% to 17%.** Training with forecast noise: 18%. NBM day-1 rain correlates 0.62 with observed; misses ~18% of rain events.
- Without any forecast, models beat persistence by only ~5%.
- Nitrate loss index (10 cm, 72 h): detecting NEW risk episodes (none in past 24 h), AUC 0.96 with perfect forecasts, **0.87 with real NBM forecasts** (persistence 0.50). It degrades far less than soil moisture, a promising paper angle. New-station AUC 0.88-0.98.
- EC (NPK-probe signal), Cook Farm: 1-3% skill over persistence at 1-3 days; within-site R² of EC ~ moisture + temp: median 0.69.
- LSTM: implemented (`soilcast/seq.py`, notebook 08, `scripts/train_lstm.py`) but **not yet run**; expect to debug on first GPU run.
- Notebooks 01-06 are executed; 07 (tuning) and 08 (LSTM) are not. The tuning code path was smoke-tested.

## Open research questions (candidate paper angles)
1. **Forecast-uncertainty-aware soil moisture prediction:** how to keep skill when rain forecasts are wrong (noise augmentation, probabilistic rain inputs, ensembles, quantile outputs).
2. **Cold-start for a new low-cost probe:** how much site data is needed; does pretraining on USCRN + SSURGO help the DFRobot probe at Jolly Lane?
3. **What low-cost NPK probes can and can't tell you,** with a bench test (distilled water → salt → fertilizer) and lab samples.
4. **Nitrate loss risk forecasting** as a farmer-facing decision tool (needs validation data to be a strong claim).
5. **Tabular GBM vs LSTM vs time-series foundation models** (TimesFM/Chronos) on sub-daily soil moisture with real forecasts.

## Repo map and commands
```
src/soilcast/  config, data (APIs), features, targets, models, evaluate, tuning, seq (LSTM), cookfarm, plotting
scripts/       download_data, build_features, tune_gbm, train_lstm, archive_nws_forecast
notebooks/     01 data, 02 relationships, 03 baselines, 04 realistic forecasts, 05 nitrate index, 06 NPK/EC, 07 tuning, 08 LSTM
tools/         make_notebooks.py (source of truth for notebooks; edit here, regenerate)
docs/          research_log.md
```
```bash
pip install -e ".[gpu-trees]"                 # into ML-env; add [deep] for torch
python scripts/build_features.py              # required once; writes data/processed (~85 MB)
python scripts/tune_gbm.py --trials 300 --storage sqlite:///results/tuning.db
python scripts/train_lstm.py --device cuda --epochs 30
python scripts/archive_nws_forecast.py --lat 44.08 --lon -103.23   # schedule hourly
```
Notebooks are generated from `tools/make_notebooks.py`. **Regenerating overwrites every notebook and wipes executed outputs**; re-execute afterward. If you edit a notebook directly, port the change back to the generator.
