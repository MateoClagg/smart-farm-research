"""Paths, stations and experiment settings shared by scripts and notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"

USER_AGENT = "sdsmt-soil-sensor-research (contact: mateo.clagg@gmail.com)"

# NOAA USCRN stations in the northern Great Plains (closest analogs to Rapid City, SD)
STATIONS = [
    "SD_Aberdeen_35_WNW", "SD_Buffalo_13_ESE", "SD_Pierre_24_S", "SD_Sioux_Falls_14_NNE",
    "WY_Sundance_8_NNW", "NE_Harrison_20_SSE", "NE_Whitman_5_ENE", "ND_Medora_7_E",
    "ND_Jamestown_38_WSW", "MT_Wolf_Point_29_ENE",
]
YEARS = range(2016, 2026)
DEPTHS_CM = (5, 10, 20)

# Splits
TRAIN_YEARS = tuple(range(2016, 2024))     # model fitting (and CV folds inside it)
TEST_YEARS = (2024, 2025)                  # touched once, at the end
CV_VALID_YEARS = (2020, 2021, 2022, 2023)  # expanding-window folds for tuning
REALISTIC_TEST_START = "2025-01-01"        # NBM forecast archive covers 2024-10-09 onward

STEP_H = 3          # issue a prediction every 3 hours
EVENT_MM = 5.0      # "rain event" = at least 5 mm in the forecast window
FROZEN_C = 0.5      # drop samples with soil temperature at or below this (probes misread ice)
SEED = 7
