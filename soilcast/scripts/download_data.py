"""Download all raw data into data/raw/ (about 5 minutes; skips files that already exist).

python scripts/download_data.py
"""
import shutil

from soilcast import data
from soilcast.config import RAW

RAW.mkdir(parents=True, exist_ok=True)

print("USCRN hourly ...")
data.download_uscrn()
u = data.parse_uscrn()
u.to_parquet(RAW / "uscrn_hourly.parquet")
coords = data.station_coords(u)
print(u.shape, "stations:", len(coords))

print("SSURGO at each station ...")
data.ssurgo_for_stations(coords).to_csv(RAW / "ssurgo_stations.csv", index=False)

print("Archived NBM forecasts (Open-Meteo Previous Runs) ...")
data.download_previous_runs(coords)
data.parse_previous_runs().to_parquet(RAW / "nbm_previous_runs.parquet")

print("Cook Agronomy Farm (GSIF) ...")
rda = data.download_cookfarm()
shutil.copy(rda, RAW / "cookfarm.rda")
print("done")
