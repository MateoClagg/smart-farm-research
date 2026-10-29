# soilcast

Soil moisture and nitrate loss risk forecasting for the SDSMT soil sensor project (Jolly Lane greenhouse prototype), developed on public data while field data accumulates.

The design follows the project spec: **previous 24 h of sensor data + NWS-style weather forecast for the next 24/72 h + SSURGO soil properties → forecast at the sensor's location and depth.**

## Setup size
- Repo zip: ~14 MB, including parsed data (`data/raw/*.parquet`). No download needed to start.
- `python scripts/build_features.py` writes ~85 MB to `data/processed/` (about 1 minute).
- Optional full re-download from source APIs (`scripts/download_data.py`): ~220 MB of raw text/JSON.
- Python packages beyond a typical ML env: `lightgbm`, `optuna`, `rdata`, `pyarrow` (and `torch` for notebook 08).

## Quick start

```bash
# Option A: existing conda env (e.g. ML-env)
conda activate ML-env
pip install -e ".[gpu-trees]"            # add ,deep if torch isn't installed yet

# Option B: fresh env
conda env create -f environment.yml && conda activate soilcast && pip install -e .

jupyter lab notebooks/
```

The repo ships with the raw data already parsed (`data/raw/*.parquet`, ~14 MB). Build the feature tables once (~1 min):

```bash
python scripts/build_features.py
```

To re-download everything from the source APIs: `python scripts/download_data.py`.

### Google Colab (A100)
1. Zip the repo folder and upload `soilcast.zip` to the root of Google Drive (MyDrive).
2. Open a notebook in Colab and set Runtime → GPU. The first cell mounts Drive, unzips, and installs.
3. Run `!python scripts/build_features.py` once per session, then the notebook.

### RTX 4080 Super PC
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match the CUDA driver
pip install -e ".[gpu-trees]"
python scripts/build_features.py
python scripts/train_lstm.py --device cuda --epochs 30
python scripts/train_lstm.py --device cuda --sweep 20                 # random search
python scripts/tune_gbm.py --kind xgb --device cuda --trials 500 --storage sqlite:///results/tuning.db
```

## Notebooks

| # | Notebook | What you learn |
|---|---|---|
| 01 | `01_data_overview` | Stations, SSURGO soils, data coverage, one season up close, how good real NBM forecasts are |
| 02 | `02_relationships` | What drives 24 h moisture change: rain dominance, exponential dry-down, soil-dependent rain response, daily cycle, depth |
| 03 | `03_baselines` | Persistence vs linear vs LightGBM across feature sets; new-station test; bootstrap CIs; SHAP |
| 04 | `04_realistic_forecasts` | Same models with **real archived NBM forecasts** (Apr to Dec 2025); training with forecast noise |
| 05 | `05_nitrate_loss_index` | The 72 h nitrate loss risk index (proposed replacement for the NPK model) |
| 06 | `06_npk_ec_check` | Why forecasting low-cost NPK probe readings reduces to forecasting EC, and how little skill that has |
| 07 | `07_tuning` | Optuna with expanding-window CV (short demo; full runs via script) |
| 08 | `08_lstm` | LSTM sequence model: smoke test on CPU, full run on GPU |

## Data

| Source | Use | Access |
|---|---|---|
| NOAA USCRN hourly, 10 northern Great Plains stations, 2016 to 2025 | Sensor inputs and targets (soil moisture/temp at 5/10/20 cm, air temp, RH, rain) | `www.ncei.noaa.gov` |
| USDA SSURGO via Soil Data Access | Sand/silt/clay, AWC, drainage class, bulk density at sensor depth | `sdmdataaccess.sc.egov.usda.gov` |
| NOAA National Blend of Models, archived (Open-Meteo Previous Runs), 2024-10-09 onward | Real forecasts at 24/48/72 h lead for the realistic test | `previous-runs-api.open-meteo.com` |
| NWS gridpoint forecast (live) | Her design's forecast input; `scripts/archive_nws_forecast.py` saves it hourly | `api.weather.gov` |
| Cook Agronomy Farm (WSU), 42 sites, daily, 2011 to 2012 | EC check for the NPK model | `github.com/cran/GSIF` |

Stations: SD Aberdeen, Buffalo, Pierre, Sioux Falls; WY Sundance; NE Harrison, Whitman; ND Medora, Jamestown; MT Wolf Point.

## Evaluation protocol

- **Time split:** train 2016 to 2023, test 2024 to 2025 (test touched once per experiment).
- **New station:** leave one station out *and* future years. Mimics a new probe at a new site.
- **Tuning:** expanding-window CV inside training years (validate 2020, 2021, 2022, 2023).
- **Realistic test:** Apr to Dec 2025 with NBM forecasts as inputs; identical rows to the perfect-forecast test.
- **Frozen soil excluded** (soil temp ≤ 0.5 °C): dielectric probes misread ice.
- Predictions issued every 3 h. Metrics in percentage points (pp) of volumetric moisture; skill = 1 − RMSE/RMSE_persistence.

## Results so far

Soil moisture 24 h ahead, 5 cm. Test 2024 to 2025, 41,813 windows (notebook 03):

| Model | RMSE (pp) | Skill vs persistence | RMSE on rain days |
|---|---|---|---|
| Persistence | 2.16 | 0 | 6.10 |
| GBM, sensors only | 2.05 | 5% | 5.82 |
| GBM, + perfect forecast + soil | 1.15 | 47% | 2.75 |
| GBM, + engineered (v2) | 1.11 [95% CI 1.01 to 1.23] | 49% | 2.60 |
| GBM v2, new station (mean of 10) | 1.36 | 30% | 3.53 |

**With real NBM forecasts** (Apr to Dec 2025, identical rows, notebook 04):

| Model | Perfect forecast | Real NBM forecast |
|---|---|---|
| Persistence RMSE | 2.40 | 2.40 |
| GBM v2 RMSE (skill) | 1.20 (50%) | 1.99 (17%) |
| GBM v2 trained with forecast noise | | 1.97 (18%) |

**Most of the "perfect forecast" skill disappears with real rain forecasts.** NBM day-1 rain correlates 0.62 with observed; it misses about 1 in 5 rain events (P(miss | rain) 0.18 in the calibration period). Rain-forecast uncertainty is the main research problem (ideas in notebooks 04 and 07).

**Nitrate loss risk index** (10 cm, next 72 h, notebook 05). Detecting new risk episodes (none in the past 24 h):

| Model | AUC, new risk |
|---|---|
| Persistence | 0.50 |
| GBM, perfect forecast | 0.95 |
| GBM, real NBM forecast | 0.87 |

The risk index holds up much better than soil moisture with real forecasts.

**NPK/EC check** (Cook Farm, notebook 06): EC forecasts beat persistence by only 1 to 3% at 1 to 3 days.

**Not yet run:** notebook 07 (tuning; the code path is smoke-tested) and notebook 08 (LSTM; run on GPU).

## Layout

```
src/soilcast/   config, data (APIs), features, targets (nitrate index), models, evaluate, tuning, seq (LSTM), cookfarm, plotting
scripts/        download_data, build_features, tune_gbm, train_lstm, archive_nws_forecast
notebooks/      01-08 (executed, with outputs)
tools/          make_notebooks.py (regenerates the notebooks)
data/raw/       parsed USCRN, SSURGO, NBM, Cook Farm
results/        CSVs written by notebooks and scripts
```

## Caveats

- **USCRN probes are research-grade**, not the low-cost DFRobot probe. Transfer to the project sensor needs calibration (oven-dried samples) and QC.
- **SSURGO is a map**, not a measurement; some readings exceed mapped porosity. Sample texture and bulk density at Jolly Lane.
- **The nitrate loss index is computed, not measured**: a decision index from published thresholds. Validation needs lab nitrate or N₂O data.
- **Greenhouse vs outdoor:** inside the greenhouse, watering replaces rain. Log every watering (time, amount) and feed it in place of the rain forecast.
