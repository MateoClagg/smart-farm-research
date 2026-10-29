"""Build model-ready feature tables into data/processed/ (a few minutes per depth).

python scripts/build_features.py            # perfect-forecast tables at 5/10/20 cm + NBM tables at 5/10 cm
"""
from soilcast import data
from soilcast.config import PROCESSED
from soilcast.features import build_features

PROCESSED.mkdir(parents=True, exist_ok=True)
u, soils = data.load_uscrn(), data.load_soils()
nbm = data.load_forecasts()

for depth in (5, 10, 20):
    print("perfect forecast, depth", depth)
    build_features(u, soils, depth).to_parquet(PROCESSED / f"features_{depth}cm_perfect.parquet")

# NBM archive starts 2024-10-09: these tables are only meaningful from then on
for depth in (5, 10):
    print("NBM forecast, depth", depth)
    f = build_features(u[u.time_utc >= "2024-09-01"], soils, depth, forecasts=nbm)
    f[f.time_utc >= "2024-10-10"].to_parquet(PROCESSED / f"features_{depth}cm_nbm.parquet")
print("done")
