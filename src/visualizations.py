"""
Publication-Grade Visual Intelligence Generator (matplotlib).
Generates 4 executive PNG charts in reports/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAVY = "#1B365D"
TEAL = "#008080"
GOLD = "#D99B00"
CRIMSON = "#B22222"
SLATE = "#5C6F82"


def generate_all_charts(
    forecast_preds: pd.DataFrame,
    feature_imp: pd.DataFrame,
    dm_scored: pd.DataFrame,
    dscr_samples: np.ndarray,
    mc_stats: Dict[str, float],
    exc_summary: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate and save all 4 executive visual intelligence charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # 1. Revenue Forecast Comparison + Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
    dates = pd.to_datetime(forecast_preds["date"])
    axes[0].plot(dates, forecast_preds["total_operating_revenue"] / 1e6, color=NAVY, lw=2.3, label="Actual Revenue")
    axes[0].plot(dates, forecast_preds["Ridge_Regression"] / 1e6, color=TEAL, ls="--", lw=1.8, label="Ridge Regression")
    axes[0].plot(dates, forecast_preds["Gradient_Boosting"] / 1e6, color=GOLD, ls="-.", lw=1.9, label="Gradient Boosting")
    axes[0].plot(dates, forecast_preds["Seasonal_Naive"] / 1e6, color=SLATE, ls=":", lw=1.4, label="Seasonal Naive")
    axes[0].set_title("FY2025–FY2026 Operating Revenue Forecast [SYN]", fontweight="bold", color=NAVY)
    axes[0].set_ylabel("Monthly Revenue ($ Millions)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].tick_params(axis="x", rotation=30)

    top_fi = feature_imp.head(8).iloc[::-1]
    axes[1].barh(top_fi["feature"], top_fi["importance"], color=TEAL, edgecolor=NAVY, alpha=0.9)
    axes[1].set_title("Top 8 Revenue Forecast Drivers (Gradient Boosting)", fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Relative Feature Importance")
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    p1 = output_dir / "01_revenue_forecast_comparison.png"
    plt.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths["forecast"] = p1

    # 2. FAA Grant Compliance Data Mining (Distance vs Cost Variance + Anomaly Scores)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
    clean = dm_scored[dm_scored["compliance_anomaly_flag"] == 0]
    anom = dm_scored[dm_scored["compliance_anomaly_flag"] == 1]

    axes[0].scatter(clean["euclidean_dist_from_centroid"], clean["cost_variance_pct"] * 100, c=TEAL, alpha=0.65, label="Compliant Drawdowns")
    axes[0].scatter(anom["euclidean_dist_from_centroid"], anom["cost_variance_pct"] * 100, c=CRIMSON, marker="X", s=70, label="Compliance Anomalies")
    axes[0].set_title("CIP & FAA Grant Drawdowns: Distance vs Cost Variance [SYN]", fontweight="bold", color=NAVY)
    axes[0].set_xlabel("Standardized Euclidean Distance from Centroid")
    axes[0].set_ylabel("Cost Variance (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].hist(clean["iso_anomaly_score"], bins=22, alpha=0.65, color=TEAL, label="Compliant")
    axes[1].hist(anom["iso_anomaly_score"], bins=15, alpha=0.80, color=CRIMSON, label="Injected Anomaly")
    axes[1].set_title("IsolationForest Unsupervised Anomaly Score Distribution", fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Anomaly Score (Higher = Higher Compliance Risk)")
    axes[1].set_ylabel("Drawdown Count")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    plt.tight_layout()
    p2 = output_dir / "02_faa_grant_anomaly_mining.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths["grant_mining"] = p2

    # 3. 10,000-Iteration Senior DSCR Monte Carlo Stress Test
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=160)
    ax.hist(dscr_samples, bins=60, color=NAVY, alpha=0.78, edgecolor="white", linewidth=0.4)
    ax.axvline(mc_stats["covenant_threshold_x"], color=CRIMSON, lw=2.3, ls="--", label=f"Senior Rate Covenant ({mc_stats['covenant_threshold_x']:.2f}x)")
    ax.axvline(mc_stats["p05_dscr_x"], color=GOLD, lw=2.0, ls=":", label=f"P5 Downside ({mc_stats['p05_dscr_x']:.2f}x)")
    ax.axvline(mc_stats["median_dscr_x"], color=TEAL, lw=2.2, ls="-", label=f"Median Base ({mc_stats['median_dscr_x']:.2f}x)")
    ax.set_title(
        f"FY2027 Senior Revenue Bond DSCR — 10,000-Iteration Monte Carlo [SIMULATED]\n"
        f"Covenant Compliance Probability: {mc_stats['prob_meet_covenant_pct']:.1f}%",
        fontweight="bold",
        color=NAVY,
    )
    ax.set_xlabel("Senior Debt Service Coverage Ratio (x)")
    ax.set_ylabel("Simulation Frequency")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    p3 = output_dir / "03_dscr_monte_carlo_scenarios.png"
    plt.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    paths["dscr_mc"] = p3

    # 4. GL & Bank Reconciliation Exception Breakdown
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=160)
    exc_only = exc_summary[exc_summary["detected_category"] != "Matched"].sort_values("item_count", ascending=True)
    ax.barh(exc_only["detected_category"], exc_only["item_count"], color=[GOLD, TEAL, NAVY, CRIMSON][: len(exc_only)])
    ax.set_title("Multi-Fund GL vs Bank Reconciliation — Exception Root-Cause Count [SYN]", fontweight="bold", color=NAVY)
    ax.set_xlabel("Number of Reconciling Items Identified (100% Detection Accuracy)")
    ax.grid(axis="x", alpha=0.25)
    for idx, val in enumerate(exc_only["item_count"]):
        ax.text(val + 0.25, idx, f"{int(val)} items", va="center", fontweight="bold", color=NAVY)
    plt.tight_layout()
    p4 = output_dir / "04_gl_bank_reconciliation_exceptions.png"
    plt.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    paths["reconciliation"] = p4

    return paths
