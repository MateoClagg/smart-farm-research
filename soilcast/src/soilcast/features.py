"""
Feature building for the project design (USCRN stations):

  past 24 h sensors : soil moisture, soil temp, air temp, RH, rain
  next 24/72 h      : rain, air temp, RH from a forecast source
                      - "perfect": observed weather (upper bound)
                      - "nbm":     archived NOAA National Blend forecasts (lead day 1/2/3)
  soil properties   : SSURGO texture, AWC, drainage class, bulk density at sensor depth
  engineered (v2)   : from the exploration (see notebooks/02_relationships.ipynb)
"""
import numpy as np
import pandas as pd

from .config import FROZEN_C, STEP_H
from .data import DRAINAGE_ORDER
from .targets import nitrate_loss_rate

# ----------------------------------------------------------------------------- feature lists
SENSORS = ["sm", "sm_l1", "sm_l3", "sm_l6", "sm_l12", "sm_l24", "dsm_1h", "dsm_6h", "dsm_24h",
           "sm_24h_mean", "sm_24h_max", "st", "st_l24", "st_24h_mean", "air_t", "air_t_24h_mean",
           "air_t_24h_max", "rh", "rh_24h_mean", "precip", "precip_6h", "precip_24h",
           "hod_sin", "hod_cos", "doy_sin", "doy_cos"]
FC24 = ["fc_precip_24h", "fc_precip_6h", "fc_air_t_mean_24h", "fc_air_t_max_24h", "fc_rh_mean_24h"]
FC72 = ["fc_precip_72h", "fc_air_t_mean_72h", "fc_air_t_max_72h", "fc_rh_mean_72h"]
SOIL = ["sand", "silt", "clay", "awc", "drainage", "bulk_density"]
ENGINEERED = ["vpd", "vpd_24h_mean", "fc_vpd_24h", "fc_et0_hargreaves", "sm_30d_min", "sm_30d_max",
              "sm_above_floor", "sm_rel_range", "drying_demand", "hours_since_rain", "storage_deficit",
              "fc_rain_x_deficit", "fc_rain_x_clay", "fc_rain_x_awc", "fc_rain_x_sand", "fc_rain_log"]
NLRI_EXTRA = ["wfps", "nl_rate_h", "nl_past24"]

FEATURE_SETS = {
    "sensors_24h": SENSORS,
    "sensors+forecast": SENSORS + FC24,
    "sensors+forecast+soil": SENSORS + FC24 + SOIL,
    "v2_engineered": SENSORS + FC24 + SOIL + ENGINEERED,
}


# ----------------------------------------------------------------------------- helpers
def _past(s, w, how):
    return getattr(s.rolling(w, min_periods=max(1, w // 2)), how)()


def _ahead(s, a, b, how):
    """Aggregate s over hours t+a .. t+b (inclusive)."""
    L = b - a + 1
    r = getattr(s[::-1].rolling(L, min_periods=max(1, L // 2)), how)()[::-1]
    return r.shift(-a)


def vpd_kpa(t_c, rh):
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    return es * (1 - np.clip(rh, 0, 100) / 100)


def extraterrestrial_rad(lat_deg, doy):
    """FAO-56 extraterrestrial radiation Ra (MJ m-2 day-1)."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta)
                                             + np.cos(phi) * np.cos(delta) * np.sin(ws))


def soil_table(soils, depth):
    s = soils[soils.depth_cm == depth].copy()
    s["drainage"] = s.drainagecl.map({k: i for i, k in enumerate(DRAINAGE_ORDER)})
    s["porosity"] = 1 - s.dbthirdbar_r / 2.65
    s = s.rename(columns={"sandtotal_r": "sand", "silttotal_r": "silt", "claytotal_r": "clay",
                          "awc_r": "awc", "dbthirdbar_r": "bulk_density"})
    return s[["station", "compname", "sand", "silt", "clay", "awc", "drainage", "bulk_density", "porosity"]]


# ----------------------------------------------------------------------------- hourly panel
def hourly_panel(uscrn, depth):
    """Complete hourly calendar per station with the sensor variables at one depth."""
    sm, st = f"SOIL_MOISTURE_{depth}", f"SOIL_TEMP_{depth}"
    d = uscrn[["station", "time_utc", "LATITUDE", sm, st, "T_HR_AVG", "RH_HR_AVG", "P_CALC"]].rename(
        columns={sm: "sm", st: "st", "T_HR_AVG": "air_t", "RH_HR_AVG": "rh", "P_CALC": "precip",
                 "LATITUDE": "lat"})
    out = []
    for s, g in d.groupby("station"):
        idx = pd.date_range(g.time_utc.min(), g.time_utc.max(), freq="h")
        g = g.set_index("time_utc").reindex(idx).rename_axis("time_utc").reset_index()
        g["station"], g["lat"] = s, g.lat.median()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _forecast_series(h, forecasts):
    """Per-hour forecast values by lead day, aligned to h (valid time). Returns dict lead -> DataFrame."""
    if forecasts is None:  # perfect forecast: every lead equals the observation
        obs = h[["precip", "air_t", "rh"]]
        return {k: obs for k in (1, 2, 3)}
    out = {}
    for k in (1, 2, 3):
        f = forecasts[forecasts.lead_day == k][["station", "time_utc", "precip", "air_t", "rh"]]
        m = h[["station", "time_utc"]].merge(f, on=["station", "time_utc"], how="left")
        out[k] = m[["precip", "air_t", "rh"]].set_index(h.index)
    return out


def add_forecast_features(h, forecasts=None):
    """fc_* features for windows t+1..t+24 (lead day 1) and t+1..t+72 (lead days 1/2/3)."""
    h = h.copy()
    fs = _forecast_series(h, forecasts)
    key = h.station
    agg = lambda frame, col, a, b, how: frame[col].groupby(key).transform(lambda s: _ahead(s, a, b, how))
    d1, d2, d3 = fs[1], fs[2], fs[3]
    h["fc_precip_24h"] = agg(d1, "precip", 1, 24, "sum")
    h["fc_precip_6h"] = agg(d1, "precip", 1, 6, "sum")
    h["fc_air_t_mean_24h"] = agg(d1, "air_t", 1, 24, "mean")
    h["fc_air_t_max_24h"] = agg(d1, "air_t", 1, 24, "max")
    h["fc_air_t_min_24h"] = agg(d1, "air_t", 1, 24, "min")
    h["fc_rh_mean_24h"] = agg(d1, "rh", 1, 24, "mean")
    parts = [(d1, 1, 24), (d2, 25, 48), (d3, 49, 72)]
    h["fc_precip_72h"] = sum(agg(f, "precip", a, b, "sum") for f, a, b in parts)
    h["fc_air_t_mean_72h"] = sum(agg(f, "air_t", a, b, "mean") for f, a, b in parts) / 3
    h["fc_air_t_max_72h"] = np.fmax.reduce([agg(f, "air_t", a, b, "max") for f, a, b in parts])
    h["fc_air_t_min_72h"] = np.fmin.reduce([agg(f, "air_t", a, b, "min") for f, a, b in parts])
    h["fc_rh_mean_72h"] = sum(agg(f, "rh", a, b, "mean") for f, a, b in parts) / 3
    return h


def add_sensor_features(h):
    h = h.copy()
    g = h.groupby("station", group_keys=False)
    for L in (1, 3, 6, 12, 24):
        h[f"sm_l{L}"] = g.sm.shift(L)
    h["dsm_1h"], h["dsm_6h"], h["dsm_24h"] = h.sm - h.sm_l1, h.sm - h.sm_l6, h.sm - h.sm_l24
    h["sm_24h_mean"] = g.sm.transform(lambda s: _past(s, 24, "mean"))
    h["sm_24h_max"] = g.sm.transform(lambda s: _past(s, 24, "max"))
    h["st_l24"] = g.st.shift(24)
    h["st_24h_mean"] = g.st.transform(lambda s: _past(s, 24, "mean"))
    h["air_t_24h_mean"] = g.air_t.transform(lambda s: _past(s, 24, "mean"))
    h["air_t_24h_max"] = g.air_t.transform(lambda s: _past(s, 24, "max"))
    h["rh_24h_mean"] = g.rh.transform(lambda s: _past(s, 24, "mean"))
    h["precip_6h"] = g.precip.transform(lambda s: _past(s, 6, "sum"))
    h["precip_24h"] = g.precip.transform(lambda s: _past(s, 24, "sum"))
    hod, doy = h.time_utc.dt.hour, h.time_utc.dt.dayofyear
    h["hod_sin"], h["hod_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    h["doy_sin"], h["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    # 30-day context and time since rain (station-agnostic, works for a new probe after ~3 days)
    h["sm_30d_min"] = g.sm.transform(lambda s: s.rolling(720, min_periods=72).min())
    h["sm_30d_max"] = g.sm.transform(lambda s: s.rolling(720, min_periods=72).max())
    last_rain = h.time_utc.where(h.precip >= 0.5).groupby(h.station).ffill()
    h["hours_since_rain"] = (h.time_utc - last_rain).dt.total_seconds() / 3600
    return h


def add_engineered(h):
    """v2 features motivated by the exploration."""
    h = h.copy()
    h["vpd"] = vpd_kpa(h.air_t, h.rh)
    h["vpd_24h_mean"] = vpd_kpa(h.air_t_24h_mean, h.rh_24h_mean)
    h["fc_vpd_24h"] = vpd_kpa(h.fc_air_t_mean_24h, h.fc_rh_mean_24h)
    ra = extraterrestrial_rad(h.lat, h.time_utc.dt.dayofyear)
    trange = (h.fc_air_t_max_24h - h.fc_air_t_min_24h).clip(lower=0)
    h["fc_et0_hargreaves"] = 0.408 * 0.0023 * ra * (h.fc_air_t_mean_24h + 17.8) * np.sqrt(trange)  # mm/day
    h["sm_above_floor"] = h.sm - h.sm_30d_min
    h["sm_rel_range"] = h.sm_above_floor / (h.sm_30d_max - h.sm_30d_min).replace(0, np.nan)
    h["drying_demand"] = h.sm_above_floor * h.fc_et0_hargreaves
    h["storage_deficit"] = h.porosity - h.sm
    r = h.fc_precip_24h.clip(upper=50)
    h["fc_rain_x_deficit"] = r * h.storage_deficit
    h["fc_rain_x_clay"] = r * h.clay
    h["fc_rain_x_awc"] = r * h.awc
    h["fc_rain_x_sand"] = r * h.sand
    h["fc_rain_log"] = np.log1p(r)
    return h


def add_targets(h):
    h = h.copy()
    g = h.groupby("station", group_keys=False)
    h["sm_t24"], h["st_t24"] = g.sm.shift(-24), g.st.shift(-24)
    h["wfps"] = h.sm / h.porosity
    h["nl_rate_h"] = nitrate_loss_rate(h.sm.to_numpy(), h.st.to_numpy(), h.porosity.to_numpy()) / 24
    g = h.groupby("station", group_keys=False)
    h["nl_past24"] = g.nl_rate_h.transform(lambda s: _past(s, 24, "sum"))
    h["nlri_72h"] = g.nl_rate_h.transform(lambda s: s[::-1].rolling(72, min_periods=72).sum()[::-1].shift(-1))
    return h


def build_features(uscrn, soils, depth, forecasts=None, step_h=STEP_H):
    """Model-ready frame, one row per station x issue time (every `step_h` hours)."""
    h = hourly_panel(uscrn, depth)
    h = h.merge(soil_table(soils, depth), on="station", how="left")
    h = add_sensor_features(h)
    h = add_forecast_features(h, forecasts)
    h = add_engineered(h)
    h = add_targets(h)
    h["year"] = h.time_utc.dt.year
    h["depth_cm"] = depth
    return h[h.time_utc.dt.hour % step_h == 0].reset_index(drop=True)


def sm_valid(d):
    """Soil moisture now and in 24 h, soil not frozen at either end."""
    return d.sm.notna() & d.sm_t24.notna() & (d.st > FROZEN_C) & (d.st_t24 > FROZEN_C)


def nlri_valid(d):
    return d.nlri_72h.notna() & d.nl_rate_h.notna() & d.porosity.notna()
