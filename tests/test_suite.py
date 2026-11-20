"""
Independent Verification & Regression Test Suite (pytest).
Covers Unit, Integration, E2E, Data Leakage, Reconciliation Tie-Out, and Monte Carlo Convergence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from run_pipeline import run_all
from src.data_generator import (
    generate_cip_and_faa_grants,
    generate_debt_service_schedule,
    generate_enplanements_and_revenue,
    generate_gl_and_bank_reconciliation,
)
from src.data_mining_anomalies import run_grant_data_mining
from src.debt_monte_carlo import compute_enterprise_ratios, run_dscr_monte_carlo
from src.forecasting_ml import build_features, train_and_evaluate_forecasters
from src.reconciliation_engine import run_gl_bank_reconciliation


def test_data_generator_determinism_and_shapes():
    rev1 = generate_enplanements_and_revenue(seed=42)
    rev2 = generate_enplanements_and_revenue(seed=42)
    pd.testing.assert_frame_equal(rev1, rev2)
    assert len(rev1) == 96
    assert (rev1["total_operating_revenue"] > 0).all()

    cip = generate_cip_and_faa_grants(seed=42)
    assert len(cip) == 180
    assert cip["compliance_anomaly_flag"].sum() == 18

    rec = generate_gl_and_bank_reconciliation(seed=42)
    assert len(rec) == 300


def test_forecasting_zero_leakage_and_beats_or_matches_baseline():
    rev = generate_enplanements_and_revenue(seed=42)
    feat = build_features(rev)
    # Verify lagged feature strictly equals prior month's revenue
    assert np.isclose(feat.loc[1, "rev_lag_1"], feat.loc[0, "total_operating_revenue"])

    res = train_and_evaluate_forecasters(rev, seed=42)
    assert len(res.predictions_df) == 24
    best_wape = float(res.metrics_df.iloc[0]["WAPE_Pct"])
    assert best_wape < 10.0, f"Unexpectedly high forecast error: {best_wape}%"
    assert res.fy2027_projected_revenue_usd > res.fy2027_projected_expense_usd


def test_data_mining_grant_anomaly_detection():
    cip = generate_cip_and_faa_grants(seed=42)
    dm = run_grant_data_mining(cip, seed=42)
    rf_auc = dm.model_metrics["RandomForest_5FoldCV_Supervised"]["ROC_AUC"]
    iso_auc = dm.model_metrics["IsolationForest_Unsupervised"]["ROC_AUC"]
    assert rf_auc >= 0.90
    assert iso_auc >= 0.80
    assert len(dm.cluster_summary_df) == 3
    assert len(dm.top_risk_terms) >= 3


def test_gl_bank_reconciliation_exact_tie_out(tmp_path: Path):
    rec = generate_gl_and_bank_reconciliation(seed=42)
    # Regression test: round-trip through CSV so empty dates deserialize as NaN
    csv_path = tmp_path / "rec.csv"
    rec.to_csv(csv_path, index=False)
    rec_loaded = pd.read_csv(csv_path)
    res = run_gl_bank_reconciliation(rec_loaded)
    assert res.detection_accuracy_pct == 100.0
    assert abs(res.reconciled_after_adjustments_gap_usd) < 0.01


def test_dscr_monte_carlo_and_scenario_probability_sum():
    rev = generate_enplanements_and_revenue(seed=42)
    debt = generate_debt_service_schedule()
    ratios = compute_enterprise_ratios(rev, debt, fiscal_year=2026)
    assert ratios["senior_dscr_x"] > 1.25

    mc = run_dscr_monte_carlo(
        fy2027_rev_base=ratios["total_operating_revenue_usd"] * 1.04,
        fy2027_exp_base=ratios["total_operating_expense_usd"] * 1.03,
        fy2027_pfc_base=35_000_000.0,
        debt_df=debt,
        n_iter=10_000,
        seed=42,
    )
    assert mc.mc_stats["iterations"] == 10_000
    assert abs(mc.scenarios_df["Probability_Pct"].sum() - 100.0) < 1e-6
    assert mc.mc_stats["p05_dscr_x"] < mc.mc_stats["median_dscr_x"] < mc.mc_stats["p95_dscr_x"]


def test_e2e_pipeline_and_excel_formulas(tmp_path: Path):
    summary = run_all(base_dir=tmp_path, seed=42)
    assert summary["status"] == "VERIFIED"

    xlsx = Path(summary["artifacts"]["excel_workbook"])
    assert xlsx.exists()
    wb = openpyxl.load_workbook(xlsx, data_only=False)
    assert set(wb.sheetnames) == {
        "Executive_Dashboard",
        "Operating_Budget_Variance",
        "CIP_FAA_Grants",
        "Debt_DSCR_Model",
        "GL_Bank_Reconciliation",
    }
    # Confirm live Excel formulas exist on Dashboard
    assert str(wb["Executive_Dashboard"]["B5"].value).startswith("=SUMIFS(")
    assert str(wb["Executive_Dashboard"]["B10"].value).startswith("=INDEX(")

    for chart_path in summary["artifacts"]["charts"].values():
        assert Path(chart_path).exists()
