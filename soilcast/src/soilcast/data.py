"""
Data access.

  USCRN hourly (NOAA NCEI)             sensors: soil moisture/temp, air temp, RH, rain
  SSURGO via Soil Data Access (USDA)   soil properties at a lat/lon and depth
  NWS gridpoint forecast (live)        her design's forecast input; archive it going forward
  Open-Meteo Previous Runs (NBM)       archived NOAA National Blend forecasts, 2024-10 onward
  Cook Agronomy Farm (WSU, via GSIF)   daily moisture/temp/EC at 42 sites (EC / NPK check)
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import RAW, STATIONS, USER_AGENT, YEARS

UA = {"User-Agent": USER_AGENT}

# ----------------------------------------------------------------------------- USCRN
USCRN_URL = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/{year}/CRNH0203-{year}-{station}.txt"
USCRN_COLS = ("WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE T_CALC T_HR_AVG "
              "T_MAX T_MIN P_CALC SOLARAD SOLARAD_FLAG SOLARAD_MAX SOLARAD_MAX_FLAG SOLARAD_MIN "
              "SOLARAD_MIN_FLAG SUR_TEMP_TYPE SUR_TEMP SUR_TEMP_FLAG SUR_TEMP_MAX SUR_TEMP_MAX_FLAG "
              "SUR_TEMP_MIN SUR_TEMP_MIN_FLAG RH_HR_AVG RH_HR_AVG_FLAG SOIL_MOISTURE_5 SOIL_MOISTURE_10 "
              "SOIL_MOISTURE_20 SOIL_MOISTURE_50 SOIL_MOISTURE_100 SOIL_TEMP_5 SOIL_TEMP_10 SOIL_TEMP_20 "
              "SOIL_TEMP_50 SOIL_TEMP_100").split()


def download_uscrn(stations=STATIONS, years=YEARS, dest=RAW / "uscrn"):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for y in years:
        for s in stations:
            f = dest / f"CRNH0203-{y}-{s}.txt"
            if f.exists():
                continue
            r = requests.get(USCRN_URL.format(year=y, station=s), headers=UA, timeout=120)
            if r.ok:
                f.write_bytes(r.content)
            else:
                print("missing", y, s, r.status_code)


def parse_uscrn(folder=RAW / "uscrn"):
    """One row per station-hour (UTC), missing codes (-9999, -99) as NaN."""
    frames = []
    for f in sorted(Path(folder).glob("CRNH0203-*.txt")):
        d = pd.read_csv(f, sep=r"\s+", header=None, names=USCRN_COLS,
                        dtype={"UTC_DATE": str, "UTC_TIME": str, "LST_DATE": str, "LST_TIME": str})
        d["station"] = f.stem.split("-", 2)[2]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["time_utc"] = pd.to_datetime(d.UTC_DATE + d.UTC_TIME.str.zfill(4), format="%Y%m%d%H%M")
    skip = {"WBANNO", "UTC_DATE", "UTC_TIME", "LST_DATE", "LST_TIME", "SUR_TEMP_TYPE"}
    num = [c for c in USCRN_COLS if not c.endswith("FLAG") and c not in skip]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce")
    meas = [c for c in num if c not in ("LONGITUDE", "LATITUDE", "CRX_VN")]
    d[meas] = d[meas].mask(d[meas] <= -99)
    keep = ["station", "time_utc", "LATITUDE", "LONGITUDE", "T_HR_AVG", "RH_HR_AVG", "P_CALC", "SOLARAD"] + \
           [c for c in USCRN_COLS if c.startswith("SOIL_")]
    return (d[keep].drop_duplicates(["station", "time_utc"])
             .sort_values(["station", "time_utc"]).reset_index(drop=True))


def load_uscrn(path=RAW / "uscrn_hourly.parquet"):
    return pd.read_parquet(path)


def station_coords(uscrn):
    return uscrn.groupby("station")[["LATITUDE", "LONGITUDE"]].median()


# ----------------------------------------------------------------------------- SSURGO
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
DRAINAGE_ORDER = ["Excessively drained", "Somewhat excessively drained", "Well drained",
                  "Moderately well drained", "Somewhat poorly drained", "Poorly drained",
                  "Very poorly drained"]


def ssurgo_at_point(lat, lon, depth_cm):
    """Dominant SSURGO component's horizon at a depth: texture, AWC, bulk density, drainage class."""
    q = f"""
    SELECT TOP 1 c.compname, c.comppct_r, c.drainagecl, h.hzdept_r, h.hzdepb_r,
           h.sandtotal_r, h.silttotal_r, h.claytotal_r, h.awc_r, h.dbthirdbar_r
    FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('point({lon} {lat})') AS m
    JOIN component AS c ON c.mukey = m.mukey
    JOIN chorizon AS h ON h.cokey = c.cokey
    WHERE h.hzdept_r <= {depth_cm} AND h.hzdepb_r > {depth_cm}
    ORDER BY c.comppct_r DESC
    """
    r = requests.post(SDA_URL, json={"query": q, "format": "JSON+COLUMNNAME"}, headers=UA, timeout=60)
    r.raise_for_status()
    t = r.json().get("Table", [])
    if len(t) < 2:
        return {}
    rec = dict(zip(t[0], t[1]))
    for k in ("comppct_r", "hzdept_r", "hzdepb_r", "sandtotal_r", "silttotal_r", "claytotal_r",
              "awc_r", "dbthirdbar_r"):
        rec[k] = float(rec[k]) if rec[k] not in (None, "") else np.nan
    return rec


def ssurgo_for_stations(coords, depths=(5, 10, 20)):
    rows = []
    for st, r in coords.iterrows():
        for dep in depths:
            rec = ssurgo_at_point(r.LATITUDE, r.LONGITUDE, dep)
            rec.update(station=st, depth_cm=dep)
            rows.append(rec)
    return pd.DataFrame(rows)


def load_soils(path=RAW / "ssurgo_stations.csv"):
    return pd.read_csv(path)


# ----------------------------------------------------------------------------- NWS (live)
def nws_forecast_grid(lat, lon):
    """Hourly NWS forecast (air_t C, rh %, precip mm) for a point. QPF split evenly over its interval."""
    pts = requests.get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}", headers=UA, timeout=30)
    pts.raise_for_status()
    grid = requests.get(pts.json()["properties"]["forecastGridData"], headers=UA, timeout=30)
    grid.raise_for_status()
    props = grid.json()["properties"]
    out = {}
    for var, split in (("temperature", False), ("relativeHumidity", False), ("quantitativePrecipitation", True)):
        rows = []
        for v in props[var]["values"]:
            start, dur = v["validTime"].split("/")
            hours = max(1, int(pd.Timedelta(dur).total_seconds() // 3600))
            idx = pd.date_range(pd.Timestamp(start), periods=hours, freq="h")
            val = v["value"] / hours if (split and v["value"] is not None) else v["value"]
            rows += [(t, val) for t in idx]
        out[var] = pd.Series(dict(rows), dtype=float)
    df = pd.DataFrame(out).rename(columns={"temperature": "air_t", "relativeHumidity": "rh",
                                           "quantitativePrecipitation": "precip"})
    df.index.name = "time_utc"
    df.attrs["issued"] = props.get("updateTime")
    return df


def archive_nws_forecast(lat, lon, folder=RAW / "nws_archive"):
    """Run on a schedule so future training uses only forecasts that existed at prediction time."""
    df = nws_forecast_grid(lat, lon)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%MZ")
    df.to_csv(folder / f"nws_{lat:.4f}_{lon:.4f}_{stamp}.csv")
    return df


# ----------------------------------------------------------------------------- Open-Meteo archived forecasts
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def download_previous_runs(coords, model="ncep_nbm_conus", start="2024-01-01", end="2025-12-31",
                           dest=RAW / "openmeteo", leads=(1, 2, 3)):
    """Archived forecasts with ~24/48/72 h lead (`*_previous_dayN`). Shared rate limit: retries with backoff."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    hourly = [f"{v}_previous_day{k}" for v in ("precipitation", "temperature_2m", "relative_humidity_2m")
              for k in leads]
    for st, r in coords.iterrows():
        f = dest / f"{model}_{st}.json"
        if f.exists():
            continue
        for attempt in range(10):
            j = requests.get(PREV_RUNS_URL, params=dict(latitude=r.LATITUDE, longitude=r.LONGITUDE,
                             hourly=",".join(hourly), start_date=start, end_date=end, models=model),
                             timeout=180).json()
            if "hourly" in j:
                f.write_text(json.dumps(j))
                break
            time.sleep(20 * (attempt + 1))


def parse_previous_runs(folder=RAW / "openmeteo", model="ncep_nbm_conus"):
    """Long frame: station, time_utc (valid time), lead_day, precip, air_t, rh."""
    rows = []
    for f in sorted(Path(folder).glob(f"{model}_*.json")):
        st = f.stem[len(model) + 1:]
        h = pd.DataFrame(json.loads(f.read_text())["hourly"])
        h["time_utc"] = pd.to_datetime(h.pop("time"))
        for k in (1, 2, 3):
            cols = {f"precipitation_previous_day{k}": "precip", f"temperature_2m_previous_day{k}": "air_t",
                    f"relative_humidity_2m_previous_day{k}": "rh"}
            if not set(cols) <= set(h.columns):
                continue
            x = h[["time_utc", *cols]].rename(columns=cols)
            x["station"], x["lead_day"] = st, k
            rows.append(x)
    return pd.concat(rows, ignore_index=True)


def load_forecasts(path=RAW / "nbm_previous_runs.parquet"):
    return pd.read_parquet(path)


# ----------------------------------------------------------------------------- Cook Agronomy Farm
def download_cookfarm(dest=RAW / "GSIF"):
    if not Path(dest).exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/cran/GSIF", str(dest)], check=True)
    return Path(dest) / "data" / "cookfarm.rda"


def load_cookfarm(rda=RAW / "cookfarm.rda"):
    """Returns dict of DataFrames: readings (daily, 42 sites x 5 depths), weather, bdensity, profiles, grids."""
    import rdata
    obj = rdata.conversion.convert(rdata.parser.parse_file(rda))["cookfarm"]
    out = {k: obj[k].copy() for k in ("readings", "weather", "bdensity", "profiles", "grids")}
    for k in ("readings", "weather"):
        out[k]["Date"] = pd.to_datetime(out[k]["Date"].astype(float), unit="D")
    out["readings"]["SOURCEID"] = out["readings"]["SOURCEID"].astype(str)
    return out
