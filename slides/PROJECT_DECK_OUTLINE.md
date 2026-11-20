# Executive Presentation Deck Outline
## ML & Financial Analytics Project for Airport Financial Analyst Senior (JR105423) — City of Austin, Dept. of Aviation

---

### Slide 1 — Title
- **Title:** Austin-Bergstrom International Airport (AUS) — Financial Analytics, Data Mining & Reconciliation Suite
- **Subtitle:** End-to-End Budgeting, FAA Grant Compliance Screening, DSCR Stress Testing & Excel 2019 Power User Model
- **Author:** Bhupesh Chandra Dimri | **Date:** November 20, 2026

### Slide 2 — Problem Overview
- **What problem does this solve?** Airport finance teams must simultaneously forecast volatile passenger-driven revenues, audit multi-million-dollar FAA CIP grant drawdowns, defend `1.25x` revenue bond covenants, and close multi-fund GL/bank reconciliations with zero discrepancies.
- **✅ Plain-English Summary:** One reproducible Python + Excel system that predicts airport revenue, flags non-compliant FAA grant invoices before submission, stress-tests bond safety across 10,000 economic futures, and reconciles every bank dollar to `$0.00`.

### Slide 3 — JD Breakdown (JR105423 & SHL Assessment)
- **Core Pillars:** Operating & CIP Budgeting · Revenue & Economic Forecasting · Airport Debt Service & Bond Compliance · FAA Grant Drawdowns · GL & Bank Reconciliations.
- **SHL Skills Assessment Alignment:** Excel 2019 Power User (dynamic formulas & formatting) · Data Mining Concepts (EDA, Euclidean distance, KMeans, TF-IDF, anomaly detection) · Financial Analysis (GASB enterprise fund ratios & DSCR).

### Slide 4 — ML & Analytical Approach
- **Problem 1 (Time-Series Regression):** Predict monthly operating revenue on a strict 24-month chronological holdout (Seasonal Naive vs Ridge vs Random Forest vs Gradient Boosting).
- **Problem 2 (Data Mining / Anomaly Detection):** Identify FAA AIP/BIL grant compliance breaches using Euclidean distance, KMeans clustering, TF-IDF text mining, IsolationForest, and 5-Fold CV RandomForest.
- **Problem 3 (Stochastic Risk & Controls):** 10,000-iteration DSCR Monte Carlo + automated multi-fund GL/bank tie-out.

### Slide 5 — Data Strategy (`[SYN]` AUS-Calibrated)
- 4 deterministic datasets (`seed=42`): 96 months of operating/enplanement history (`FY2019–FY2026`), 180 CIP/FAA grant drawdowns (`10%` injected compliance defects), 300 multi-fund GL/bank lines (`4` enterprise funds), and 10-year revenue bond debt schedule.
- Leakage-free preprocessing: strictly lagged features (`t-1, t-3, t-12`), cyclical month encoding, explicit null-date handling.

### Slide 6 — Model Design & Engineering Discovery
- **Forecasting Winner:** Ridge Regression (`4.74% WAPE`, `4.88% MAPE`) beat Seasonal Naive (`7.12%`) and tree models (`5.77%–6.54%`) due to clean linear trend extrapolation.
- **Data Mining Discovery:** Isolating dense numeric drivers from sparse TF-IDF features raised unsupervised `IsolationForest ROC-AUC` from `0.7682` to `0.9880` (`PR-AUC 0.9271`).

### Slide 7 — Results & Visual Insights
- **FY2026 Enterprise Baseline:** `$382.13M` Revenue · `44.63%` Operating Margin · `$11.96` CPE · `2.38x` Senior DSCR · `422.6` Days Cash on Hand.
- **GL & Bank Reconciliation:** `100.0%` detection of 48 exceptions (`$4.40M` raw variance) -> **`$0.00` post-adjustment gap**.
- **DSCR Monte Carlo:** `2.137x` Median · `1.512x` P5 Downside · `99.55%` probability of clearing `1.25x` Senior Rate Covenant.

### Slide 8 — Business Impact for Austin Aviation
- **Faster Budget Season:** Automated dynamic Excel 2019 workbook (`SUMIFS`, `INDEX/MATCH`, conditional formatting) cuts manual variance compilation.
- **Audit-Ready FAA Compliance:** Pre-screens federal-share cap breaches (`>75%`) and duplicate invoices before federal drawdowns.
- **Bond Rating Protection:** Quantifies exact enplanement and O&M thresholds before covenant headroom narrows.

### Slide 9 — Future Improvements (30/60/90-Day Production Roadmap)
- Connect directly to City of Austin ACFR & BTS T-100 enplanement feeds.
- Expand Monte Carlo to multi-year FY2027–FY2032 CIP debt-issuance sizing.
- Add automated Power BI / Streamlit interactive variance drill-down for Aviation Executive Leadership.

### Slide 10 — About This Project
- **Created by:** Bhupesh Chandra Dimri
- **Repository:** `https://github.com/bhupesh-debug/austin-aviation-financial-analytics`
- **Contact:** `Bhupesh7cd@gmail.com` | `917-818-6198`
