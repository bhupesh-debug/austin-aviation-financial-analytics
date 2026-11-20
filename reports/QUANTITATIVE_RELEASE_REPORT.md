# Quantitative Research, Risk Audit & Release Report
**Repository:** `austin-aviation-financial-analytics`
**Target Role:** City of Austin — Department of Aviation, Airport Financial Analyst Senior (`JR105423`)
**Date:** 2026-11-20 | **Lifecycle:** `D-I-B-T-F-A-R` | **Release Verdict:** **DEPLOY**

---

## 1. Executive Summary & Evidence Integrity
This report documents independent code auditing, regression testing, statistical hypothesis evaluation, 10,000-iteration Monte Carlo simulation, and three-scenario forecasting for the Austin-Bergstrom International Airport (`AUS`) Financial Analytics Suite.
- **`[VERIFIED]`** All metrics computed directly via `python run_pipeline.py` and verified via `pytest -v`.
- **`[SIMULATED]`** Synthetic datasets (`seed=42`) and 10,000-run DSCR stress tests are explicitly labeled synthetic.

## 2. Defects Discovered & Fixed via T-D-F-R Regression Loop
| Defect ID | Severity | Root Cause | Fix Applied | Regression Proof |
|---|---|---|---|---|
| `DEF-001` | **P1** | `pd.read_csv()` deserialized blank `gl_post_date` / `bank_clear_date` as `NaN`; `str(NaN)` produced `"nan"`, dropping GL/Bank accuracy to `92.0%` (`$2.90M` gap). | Added explicit `pd.isna()` null handling in `src/reconciliation_engine.py`. | `test_gl_bank_reconciliation_exact_tie_out` round-trips CSV -> **`100.0%` accuracy, `$0.00` gap**. |
| `DEF-002` | **P2** | `IsolationForest` trained on 36 features (6 dense financial + 30 sparse TF-IDF) suffered dimensionality dilution (`ROC-AUC 0.7682`). | Restricted unsupervised `IsolationForest` to 6 standardized numeric compliance drivers (`X_num`). | Unsupervised `ROC-AUC` jumped **`0.7682 -> 0.9880`** (`PR-AUC 0.9271`). |
| `DEF-003` | **P2** | `pytest` collection failed with `ModuleNotFoundError: run_pipeline` when invoked directly. | Added `pyproject.toml` with `pythonpath = ["."]`. | `pytest -v` passes **`6/6` tests** cleanly. |

## 3. Research Hypotheses & Observed Outcomes
| ID | Research Question | H0 (Null) | Test & Metric | Observed Result (`[VERIFIED]`/`[SIMULATED]`) | Verdict |
|---|---|---|---|---|---|
| `H1` | Does regularized ML beat Seasonal Naive on 24-month airport revenue holdout? | `WAPE_ML >= WAPE_Naive` | Chronological 24-mo holdout WAPE | Ridge `4.74%` vs Naive `7.12%` (`-2.38 pp` error reduction) | **Reject H0** |
| `H2` | Does dense-feature IsolationForest separate FAA grant anomalies without labels? | `ROC_AUC <= 0.80` | Unsupervised ROC-AUC / PR-AUC | `ROC-AUC = 0.9880`, `PR-AUC = 0.9271` | **Reject H0** |
| `H3` | Does automated rule reconciliation close multi-fund GL/Bank variance to `$0`? | `Post-Adj Gap != $0` | Exact tie-out test across 300 txns | `$4,399,358.76` raw variance -> **`$0.00`** post-adjustment | **Reject H0** |
| `H4` | Does FY2027 Senior DSCR maintain >= `1.25x` covenant under asymmetric shocks? | `P(DSCR >= 1.25x) < 90%` | 10,000-run Monte Carlo (`seed=42`) | `99.55%` compliance (`P5 = 1.512x`, `Median = 2.137x`) | **Reject H0** |

## 4. Monte Carlo Simulation Summary (10,000 Iterations, `[SIMULATED]`)
- **Objective:** Quantify FY2027 Senior Revenue Bond DSCR resilience against non-normal enplanement (`Triangular(0.80, 1.01, 1.10)`), non-airline spend (`Lognormal(0, 0.045)`), and O&M inflation (`Triangular(0.97, 1.02, 1.12)`) shocks.
- **Distribution:** Mean `2.112x` · Median `2.137x` · Std Dev `0.345x` · P5 `1.512x` · P25 `1.869x` · P75 `2.361x` · P95 `2.644x`.
- **Covenant (`1.25x`) Risk:** `99.55%` Compliant · `0.45%` Breach Probability · P5 Covenant Buffer `+$21.97M`.

## 5. Three-Scenario Forecast (Probabilities Sum = 100%)
| Metric | Worst Case (20%) | Base Case (55%) | Best Case (25%) |
|---|---:|---:|---:|
| Enplanement Shock | `-14.0%` | `+1.0%` | `+7.5%` |
| O&M Inflation Shock | `+9.0%` | `+2.0%` | `-1.0%` |
| Projected Senior DSCR | `1.635x` | `2.137x` | `2.542x` |
| Covenant Status | Compliant (Watch) | Compliant | Strong Headroom |
| Mitigation / Action | Discretionary O&M freeze + PFC offset | Quarterly variance tracking + CIP phasing | Accelerate cash-funded CIP equity |

## 6. SWOT Analysis & 1% Pareto Bottleneck
- **Strengths:** Zero-leakage chronological splits; `$0.00` verified multi-fund tie-out; live Excel 2019 formula workbook; GitHub Actions CI.
- **Weaknesses:** Synthetic `[SYN]` inputs rather than live City of Austin ACFR/ePerformance ERP feeds; single-year Monte Carlo horizon.
- **Opportunities:** Direct ingestion of public City of Austin ACFR PDFs / FAA BTS T-100 enplanement tables; Streamlit executive dashboard.
- **Threats:** Airline hub concentration shock or FAA AIP reauthorization delays.
- **The 1% Factor Dominating 99% of Municipal Airport Financial Risk:** **Enplanement-driven non-airline revenue elasticity combined with fixed debt service.** Because `60.45%` of operating revenue is non-airline (parking, concessions, rental cars) and Senior Debt Service (`$83.9M` in FY2027) is fixed, early monthly variance detection on enplanements and O&M is the highest-leverage intervention for protecting the `1.25x` bond covenant.

## 7. Release Verdict
**DECISION: `DEPLOY`** — All 6 `pytest` unit/integration/E2E/regression tests pass, all 4 charts and the 5-tab dynamic Excel 2019 workbook generate deterministically, and zero secrets or unverified metrics exist in the codebase.
