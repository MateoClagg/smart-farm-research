"""
Nitrate loss risk index (heuristic, built from published numbers; not measured N loss).

  wetness factor  : 0 below 60% water-filled pore space (WFPS), 1 at >= 80%, linear between.
                    Denitrification rises sharply at 60-80% WFPS (HESS 8:686, 2004).
  temperature rate: % of soil nitrate lost per saturated day (Purdue Extension):
                    < 55 F: 1-2, 55-65 F: 2-3, > 65 F: 4-5. Midpoints used.

daily rate (%/day) = wetness factor x temperature rate
"""
import numpy as np

PARTICLE_DENSITY = 2.65
WFPS_ON, WFPS_FULL = 0.60, 0.80
TEMP_BREAKS_C = (12.8, 18.3)  # 55 F, 65 F
TEMP_RATES = (1.5, 2.5, 4.5)


def porosity_from_bulk_density(bd):
    return 1 - bd / PARTICLE_DENSITY


def nitrate_loss_rate(vw, st, porosity):
    """Daily % of soil nitrate at risk (NaN where inputs missing)."""
    vw, st, porosity = map(np.asarray, (vw, st, porosity))
    wet = np.clip((vw / porosity - WFPS_ON) / (WFPS_FULL - WFPS_ON), 0, 1)
    rate = np.select([st < TEMP_BREAKS_C[0], st < TEMP_BREAKS_C[1]], TEMP_RATES[:2], TEMP_RATES[2])
    out = wet * rate
    return np.where(np.isnan(vw) | np.isnan(st) | np.isnan(porosity), np.nan, out)
