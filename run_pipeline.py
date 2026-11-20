"""
End-to-End Orchestrator for Austin-Bergstrom International Airport (AUS)
Financial Analytics, Data Mining, Debt Stress-Testing & Reconciliation Suite.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.data_generator import generate_all_datasets
from src.data_mining_anomalies import run_grant_data_mining
from src.debt_monte_carlo import compute_enterprise_ratios, run_dscr_monte_carlo
from src.excel_power_user_builder import build_excel_power_user_workbook
from src.forecasting_ml import train_and_evaluate_forecasters
from src.reconciliation_engine import run_gl_bank_reconciliation
from src.visualizations import generate_all_charts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_all(base_dir: Path | None = None, seed: int = 42) -> Dict[str, Any]:
    """Execute full pipeline and persist data, Excel workbook, charts, and JSON metrics."""
    root = base_dir or Path(__file__).resolve().parent
    data_dir = root / "data" / "synthetic"
    reports_dir = root / "reports"
    fig_dir = reports_dir / "figures"

    # 1. Generate synthetic AUS datasets
    files = generate_all_datasets(data_dir, seed=seed)
    rev_df = pd.read_csv(files["enplanements_and_revenue"])
    cip_df = pd.read_csv(files["cip_projects_and_faa_grants"])
    rec_df = pd.read_csv(files["gl_and_bank_reconciliation"])
    debt_df = pd.read_csv(files["debt_service_schedule"])

    # 2. ML Operating Revenue Forecasting
    fc_res = train_and_evaluate_forecasters(rev_df, seed=seed)

    # 3. SHL Data Mining & FAA Grant Compliance Anomaly Detection
    dm_res = run_grant_data_mining(cip_df, seed=seed)

    # 4. Multi-Fund GL & Bank Reconciliation
    rec_res = run_gl_bank_reconciliation(rec_df)

    # 5. GASB Enterprise Ratios & 10,000-Iteration DSCR Monte Carlo
    fy26_ratios = compute_enterprise_ratios(rev_df, debt_df, fiscal_year=2026)
    fy26_pfc = float(rev_df[rev_df["fiscal_year"] == 2026]["pfc_revenue"].sum())
    mc_res = run_dscr_monte_carlo(
        fy2027_rev_base=fc_res.fy2027_projected_revenue_usd,
        fy2027_exp_base=fc_res.fy2027_projected_expense_usd,
        fy2027_pfc_base=fy26_pfc * 1.03,
        debt_df=debt_df,
        n_iter=10_000,
        seed=seed,
    )
    mc_res.fy2026_ratios = fy26_ratios

    # 6. Build Excel 2019 Power User Model (.xlsx)
    xlsx_path = build_excel_power_user_workbook(
        rev_df=rev_df,
        cip_df=dm_res.scored_df,
        debt_df=debt_df,
        rec_df=rec_res.reconciled_df,
        mc_stats=mc_res.mc_stats,
        output_path=reports_dir / "AUS_Aviation_Financial_Model_Excel2019.xlsx",
    )

    # 7. Generate Visual Intelligence Charts
    chart_paths = generate_all_charts(
        forecast_preds=fc_res.predictions_df,
        feature_imp=fc_res.feature_importance_df,
        dm_scored=dm_res.scored_df,
        dscr_samples=mc_res.dscr_samples,
        mc_stats=mc_res.mc_stats,
        exc_summary=rec_res.exception_summary_df,
        output_dir=fig_dir,
    )

    summary = {
        "status": "VERIFIED",
        "forecast_best_model": fc_res.best_model_name,
        "forecast_metrics": fc_res.metrics_df.to_dict(orient="records"),
        "fy2027_budget_projection": {
            "revenue_usd": fc_res.fy2027_projected_revenue_usd,
            "expense_usd": fc_res.fy2027_projected_expense_usd,
            "noi_usd": fc_res.fy2027_projected_noi_usd,
        },
        "data_mining_metrics": dm_res.model_metrics,
        "top_risk_terms": dm_res.top_risk_terms,
        "reconciliation": {
            "detection_accuracy_pct": rec_res.detection_accuracy_pct,
            "total_gl_usd": rec_res.total_gl_usd,
            "total_bank_usd": rec_res.total_bank_usd,
            "raw_variance_usd": rec_res.net_variance_usd,
            "post_adjustment_gap_usd": rec_res.reconciled_after_adjustments_gap_usd,
            "exceptions_by_type": rec_res.exception_summary_df.to_dict(orient="records"),
        },
        "fy2026_enterprise_ratios": fy26_ratios,
        "dscr_monte_carlo": mc_res.mc_stats,
        "three_scenarios": mc_res.scenarios_df.to_dict(orient="records"),
        "artifacts": {
            "excel_workbook": str(xlsx_path),
            "charts": {k: str(v) for k, v in chart_paths.items()},
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "pipeline_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Pipeline complete. Metrics written to %s", metrics_path)
    return summary


if __name__ == "__main__":
    result = run_all()
    print(json.dumps({
        "best_forecast_model": result["forecast_best_model"],
        "fy2026_senior_dscr": result["fy2026_enterprise_ratios"]["senior_dscr_x"],
        "mc_median_dscr": result["dscr_monte_carlo"]["median_dscr_x"],
        "mc_covenant_prob_pct": result["dscr_monte_carlo"]["prob_meet_covenant_pct"],
        "rf_grant_roc_auc": result["data_mining_metrics"]["RandomForest_5FoldCV_Supervised"]["ROC_AUC"],
        "gl_bank_detection_acc_pct": result["reconciliation"]["detection_accuracy_pct"],
        "gl_bank_post_adj_gap_usd": result["reconciliation"]["post_adjustment_gap_usd"],
    }, indent=2))
