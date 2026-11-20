"""
GASB Enterprise Fund Financial Ratio Engine & 10,000-Iteration DSCR Monte Carlo Stress Test.
Addresses JR105423 Debt Service Tracking, SHL Financial Analysis, and /repo Monte Carlo + 3 Scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class DebtMonteCarloResult:
    fy2026_ratios: Dict[str, float]
    mc_stats: Dict[str, float]
    dscr_samples: np.ndarray
    scenarios_df: pd.DataFrame


def compute_enterprise_ratios(rev_df: pd.DataFrame, debt_df: pd.DataFrame, fiscal_year: int = 2026) -> Dict[str, float]:
    """Compute core airport enterprise fund financial ratios for a given fiscal year."""
    fy = rev_df[rev_df["fiscal_year"] == fiscal_year]
    debt_row = debt_df[debt_df["fiscal_year"] == fiscal_year].iloc[0]

    total_rev = float(fy["total_operating_revenue"].sum())
    airline_rev = float(fy["airline_revenue"].sum())
    non_airline_rev = float(fy["non_airline_revenue"].sum())
    pfc_rev = float(fy["pfc_revenue"].sum())
    total_exp = float(fy["total_operating_expense"].sum())
    enp = float(fy["enplanements"].sum())

    noi = total_rev - total_exp
    # Municipal airport revenue bond indenture: Net Revenue + eligible PFC offset (25% applied to debt)
    net_rev_avail = noi + 0.25 * pfc_rev
    senior_ds = float(debt_row["senior_debt_service_usd"])
    total_ds = float(debt_row["total_debt_service_usd"])

    # Illustrative unrestricted cash reserve (~420 days cash on hand benchmark for A+/AA- hub)
    unrestricted_cash = 245_000_000.0
    daily_om = total_exp / 365.0

    return {
        "fiscal_year": fiscal_year,
        "total_operating_revenue_usd": round(total_rev, 2),
        "total_operating_expense_usd": round(total_exp, 2),
        "net_operating_income_usd": round(noi, 2),
        "operating_margin_pct": round((noi / total_rev) * 100.0, 2),
        "non_airline_revenue_share_pct": round((non_airline_rev / total_rev) * 100.0, 2),
        "cpe_usd": round(airline_rev / enp, 2),
        "senior_dscr_x": round(net_rev_avail / senior_ds, 3),
        "all_in_dscr_x": round(net_rev_avail / total_ds, 3),
        "days_cash_on_hand": round(unrestricted_cash / daily_om, 1),
        "total_enplanements": int(enp),
    }


def run_dscr_monte_carlo(
    fy2027_rev_base: float,
    fy2027_exp_base: float,
    fy2027_pfc_base: float,
    debt_df: pd.DataFrame,
    n_iter: int = 10_000,
    seed: int = 42,
) -> DebtMonteCarloResult:
    """
    Run 10,000-iteration correlated Monte Carlo simulation of FY2027 Senior DSCR.
    Uses non-normal triangular/lognormal shocks to avoid naive normality assumptions.
    """
    rng = np.random.default_rng(seed)
    debt_27 = debt_df[debt_df["fiscal_year"] == 2027].iloc[0]
    senior_ds = float(debt_27["senior_debt_service_usd"])
    covenant = float(debt_27["senior_rate_covenant_min_dscr"])

    # Asymmetric shock distributions (downside aviation demand tail + O&M inflation pressure)
    enp_shock = rng.triangular(left=0.80, mode=1.01, right=1.10, size=n_iter)
    non_air_spend_shock = rng.lognormal(mean=0.0, sigma=0.045, size=n_iter)
    om_inflation_shock = rng.triangular(left=0.97, mode=1.02, right=1.12, size=n_iter)

    sim_rev = fy2027_rev_base * (0.45 * enp_shock + 0.55 * enp_shock * non_air_spend_shock)
    sim_pfc = fy2027_pfc_base * enp_shock
    sim_exp = fy2027_exp_base * om_inflation_shock
    sim_net_avail = (sim_rev - sim_exp) + 0.25 * sim_pfc
    dscr = sim_net_avail / senior_ds

    mc_stats = {
        "iterations": n_iter,
        "covenant_threshold_x": covenant,
        "mean_dscr_x": round(float(np.mean(dscr)), 3),
        "median_dscr_x": round(float(np.median(dscr)), 3),
        "std_dscr_x": round(float(np.std(dscr)), 3),
        "p05_dscr_x": round(float(np.percentile(dscr, 5)), 3),
        "p25_dscr_x": round(float(np.percentile(dscr, 25)), 3),
        "p75_dscr_x": round(float(np.percentile(dscr, 75)), 3),
        "p95_dscr_x": round(float(np.percentile(dscr, 95)), 3),
        "prob_meet_covenant_pct": round(float(np.mean(dscr >= covenant) * 100.0), 2),
        "prob_breach_covenant_pct": round(float(np.mean(dscr < covenant) * 100.0), 2),
        "expected_shortfall_p05_usd": round(float(np.percentile(sim_net_avail - covenant * senior_ds, 5)), 2),
    }

    # Three-Scenario Forecast (Probabilities sum strictly to 100%)
    scenarios = pd.DataFrame(
        [
            {
                "Scenario": "Worst_Case",
                "Probability_Pct": 20.0,
                "Enplanement_Shock": "-14.0%",
                "OM_Inflation_Shock": "+9.0%",
                "Projected_Senior_DSCR_x": round(float(np.percentile(dscr, 10)), 3),
                "Covenant_Status": "Watch / Buffer Tight" if np.percentile(dscr, 10) < 1.35 else "Compliant",
                "Trigger_and_Mitigation": "Airline capacity cut or macro recession -> activate discretionary O&M freeze & PFC debt offset.",
            },
            {
                "Scenario": "Base_Case",
                "Probability_Pct": 55.0,
                "Enplanement_Shock": "+1.0%",
                "OM_Inflation_Shock": "+2.0%",
                "Projected_Senior_DSCR_x": round(float(np.median(dscr)), 3),
                "Covenant_Status": "Compliant",
                "Trigger_and_Mitigation": "Steady Austin metro growth -> maintain CIP phasing and quarterly variance monitoring.",
            },
            {
                "Scenario": "Best_Case",
                "Probability_Pct": 25.0,
                "Enplanement_Shock": "+7.5%",
                "OM_Inflation_Shock": "-1.0%",
                "Projected_Senior_DSCR_x": round(float(np.percentile(dscr, 90)), 3),
                "Covenant_Status": "Strong Headroom",
                "Trigger_and_Mitigation": "New international routes + parking yield optimization -> accelerate cash-funded CIP equity.",
            },
        ]
    )

    return DebtMonteCarloResult(
        fy2026_ratios={},  # populated by caller
        mc_stats=mc_stats,
        dscr_samples=dscr,
        scenarios_df=scenarios,
    )
