"""Save the current NWS gridpoint forecast for the field site (run hourly via cron / Task Scheduler).

Building this archive now means future models can be trained on the forecasts that actually
existed at prediction time (her design's "save forecasts when issued").

  python scripts/archive_nws_forecast.py --lat 44.08 --lon -103.23
"""
import argparse

from soilcast.data import archive_nws_forecast

ap = argparse.ArgumentParser()
ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--lon", type=float, required=True)
a = ap.parse_args()
df = archive_nws_forecast(a.lat, a.lon)
print(df.attrs.get("issued"), len(df), "hours saved")
