# Research log

One dated entry per decision or result worth keeping. Format: **what / why / result / next**. Include negative results.

## 2026-10-28: baseline setup
- **What:** Built soil moisture (24 h) and nitrate loss index (72 h) baselines on USCRN (10 stations, 2016-2025) + SSURGO; NPK question checked on Cook Farm EC.
- **Why:** The Jolly Lane prototype has ~4 days of usable data; NPK and rain sensors read 0.
- **Result:** GBM + perfect forecast + engineered features: 49% skill over persistence at 5 cm. With real NBM forecasts: 17%. EC (the signal behind low-cost NPK probes) barely beats persistence.
- **Decision (pending with team):** Replace the NPK forecast target with the nitrate loss risk index; keep NPK/EC as a "pull a soil test" flag.
- **Next:** Confirm venue/deadline with Dr. Mehrabian. Mateo picks the paper angle (see CLAUDE.md "Open research questions"). Start archiving NWS forecasts for Jolly Lane. Bench-test the NPK probe.
