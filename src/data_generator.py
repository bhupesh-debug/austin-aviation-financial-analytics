"""
Synthetic [SYN] Municipal Aviation Financial Dataset Generator
Calibrated to Austin-Bergstrom International Airport (AUS) Enterprise Fund Structure.

Generates 4 deterministic datasets (seed=42):
1. enplanements_and_revenue.csv  (96 monthly periods, FY2019-FY2026)
2. cip_projects_and_faa_grants.csv (180 CIP / FAA AIP & BIL grant drawdowns)
3. gl_and_bank_reconciliation.csv (600 multi-fund GL & bank statement lines)
4. debt_service_schedule.csv     (10-year Senior & Subordinate Revenue Bond schedule)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_enplanements_and_revenue(seed: int = 42) -> pd.DataFrame:
    """Generate 96 months (FY2019-FY2026) of AUS-style operating & revenue metrics."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-10-01", periods=96, freq="MS")  # Municipal FY starts Oct 1

    rows = []
    for i, dt in enumerate(dates):
        fy = dt.year + 1 if dt.month >= 10 else dt.year
        month = dt.month

        # Seasonal factor: Austin peaks in Mar (SXSW), Jun-Jul (summer), Oct (F1/ACL), Nov-Dec (holidays)
        seasonality = {
            1: 0.88, 2: 0.86, 3: 1.09, 4: 1.02, 5: 1.05, 6: 1.12,
            7: 1.13, 8: 1.01, 9: 0.96, 10: 1.10, 11: 1.03, 12: 1.06,
        }[month]

        # Structural growth trend + COVID shock (FY2020-FY2021) + Austin tech boom recovery
        base_enp = 680_000 * ((1 + 0.055 / 12) ** i)
        if dt >= pd.Timestamp("2020-03-01") and dt <= pd.Timestamp("2020-06-01"):
            shock = 0.22
        elif dt >= pd.Timestamp("2020-07-01") and dt <= pd.Timestamp("2021-03-01"):
            shock = 0.48 + 0.04 * ((dt.year - 2020) * 12 + dt.month - 7)
        elif dt >= pd.Timestamp("2021-04-01") and dt <= pd.Timestamp("2021-12-01"):
            shock = 0.88 + 0.015 * (dt.month - 4)
        else:
            shock = 1.0

        austin_metro_emp_idx = 100.0 + 0.32 * i + rng.normal(0, 0.8)
        jet_fuel_price_gal = float(np.clip(2.10 + 0.008 * i + 0.35 * np.sin(i / 6.0) + rng.normal(0, 0.12), 1.20, 4.10))
        cpi_index = 252.0 * ((1 + 0.032 / 12) ** i)

        enplanements = int(max(95_000, base_enp * seasonality * shock * rng.normal(1.0, 0.025)))
        landed_weight_klbs = round(enplanements * rng.normal(1.18, 0.02), 1)

        # Airline revenues (landing fees + terminal space rentals)
        landing_fee_rate = 3.85 * ((1 + 0.025 / 12) ** i)
        landing_fee_rev = round(landed_weight_klbs * landing_fee_rate, 2)
        terminal_rental_rev = round((5_200_000 + 18_500 * i) * rng.normal(1.0, 0.015), 2)
        airline_revenue = round(landing_fee_rev + terminal_rental_rev, 2)

        # Non-airline revenues (parking, rental cars, food/bev concessions)
        spend_per_enp = 14.80 * ((1 + 0.028 / 12) ** i) * rng.normal(1.0, 0.03)
        parking_rev = round(enplanements * spend_per_enp * 0.46, 2)
        concessions_rev = round(enplanements * spend_per_enp * 0.32, 2)
        rental_car_rev = round(enplanements * spend_per_enp * 0.22, 2)
        non_airline_revenue = round(parking_rev + concessions_rev + rental_car_rev, 2)

        # Passenger Facility Charge (PFC) - $4.50 statutory cap, ~88% collection rate
        pfc_revenue = round(enplanements * 4.50 * 0.88, 2)

        total_operating_revenue = round(airline_revenue + non_airline_revenue, 2)

        # Operating Expenses (Personnel, Contractual/Utilities, Maintenance)
        personnel_exp = round((6_800_000 + 31_000 * i) * rng.normal(1.0, 0.018), 2)
        contractual_exp = round((3_900_000 + 16_000 * i) * seasonality * rng.normal(1.0, 0.025), 2)
        maintenance_exp = round((1_750_000 + 8_500 * i) * rng.normal(1.0, 0.04), 2)
        total_operating_expense = round(personnel_exp + contractual_exp + maintenance_exp, 2)

        # Adopted Budget benchmark (set prior to FY with ~3-5% planning error)
        budget_revenue = round(total_operating_revenue * rng.normal(0.985, 0.035), 2)
        budget_expense = round(total_operating_expense * rng.normal(1.01, 0.028), 2)

        net_operating_income = round(total_operating_revenue - total_operating_expense, 2)
        cpe = round(airline_revenue / enplanements, 2)

        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "fiscal_year": fy,
                "month": month,
                "enplanements": enplanements,
                "landed_weight_klbs": landed_weight_klbs,
                "austin_metro_emp_idx": round(austin_metro_emp_idx, 2),
                "jet_fuel_price_gal": round(jet_fuel_price_gal, 2),
                "cpi_index": round(cpi_index, 2),
                "landing_fee_rev": landing_fee_rev,
                "terminal_rental_rev": terminal_rental_rev,
                "airline_revenue": airline_revenue,
                "parking_rev": parking_rev,
                "concessions_rev": concessions_rev,
                "rental_car_rev": rental_car_rev,
                "non_airline_revenue": non_airline_revenue,
                "pfc_revenue": pfc_revenue,
                "total_operating_revenue": total_operating_revenue,
                "budget_revenue": budget_revenue,
                "personnel_exp": personnel_exp,
                "contractual_exp": contractual_exp,
                "maintenance_exp": maintenance_exp,
                "total_operating_expense": total_operating_expense,
                "budget_expense": budget_expense,
                "net_operating_income": net_operating_income,
                "cpe_usd": cpe,
            }
        )

    return pd.DataFrame(rows)


def generate_cip_and_faa_grants(seed: int = 42) -> pd.DataFrame:
    """Generate 180 CIP project drawdown & FAA AIP/BIL grant compliance records."""
    rng = np.random.default_rng(seed + 1)
    programs = ["Airfield & Runway", "Terminal Expansion", "Baggage Handling", "Apron & Taxiway", "Utility & IT"]
    grant_types = ["FAA_AIP_Entitlement", "FAA_AIP_Discretionary", "FAA_BIL_ATP", "Local_Revenue_Bond"]
    narratives_clean = [
        "Runway 18L/36R asphalt overlay progress billing certified by resident engineer",
        "Concourse B gate expansion structural steel milestone drawdown verified",
        "Automated baggage screening EDS installation phase 2 FAA eligible invoice",
        "Taxiway Bravo LED lighting upgrade federal share reimbursement request",
        "Stormwater drainage apron rehabilitation certified contractor pay app",
    ]
    narratives_anomalous = [
        "Urgent duplicate resubmission prior period invoice without lien waiver",
        "Lobby executive lounge furniture and non-aeronautical artwork ineligible cost",
        "Manual override federal share exceeded 75 percent statutory cap adjustment",
        "Unapproved change order vendor surcharge missing FAA concurrence memo",
    ]

    rows = []
    for idx in range(1, 181):
        program = programs[idx % len(programs)]
        grant_type = grant_types[int(rng.integers(0, len(grant_types)))]
        budget_alloc = float(np.round(rng.uniform(1_500_000, 28_000_000), 2))
        drawdown_amt = float(np.round(budget_alloc * rng.uniform(0.05, 0.22), 2))
        # Large-hub FAA AIP statutory federal share is typically 75% (0% for Local Bond)
        fed_share_cap = 0.0 if grant_type == "Local_Revenue_Bond" else 0.75
        claimed_fed_share = fed_share_cap if fed_share_cap == 0 else float(np.round(rng.uniform(0.60, 0.75), 3))
        retainage_pct = float(np.round(rng.uniform(0.05, 0.10), 3))
        days_since_invoice = int(rng.integers(5, 45))
        cost_variance_pct = float(np.round(rng.normal(0.02, 0.04), 4))
        narrative = narratives_clean[int(rng.integers(0, len(narratives_clean)))]
        is_anomaly = 0

        # Inject 18 realistic compliance anomalies (10% rate)
        if idx % 10 == 0:
            is_anomaly = 1
            anomaly_kind = idx % 4
            if anomaly_kind == 0:
                claimed_fed_share = float(np.round(rng.uniform(0.82, 0.96), 3))
                narrative = narratives_anomalous[2]
            elif anomaly_kind == 1:
                cost_variance_pct = float(np.round(rng.uniform(0.24, 0.48), 4))
                narrative = narratives_anomalous[3]
            elif anomaly_kind == 2:
                days_since_invoice = int(rng.integers(95, 165))
                retainage_pct = 0.0
                narrative = narratives_anomalous[0]
            else:
                drawdown_amt = float(np.round(budget_alloc * rng.uniform(0.32, 0.45), 2))
                narrative = narratives_anomalous[1]

        rows.append(
            {
                "drawdown_id": f"CIP-2026-{idx:04d}",
                "program": program,
                "grant_type": grant_type,
                "project_budget_usd": budget_alloc,
                "drawdown_amount_usd": drawdown_amt,
                "drawdown_to_budget_ratio": round(drawdown_amt / budget_alloc, 4),
                "federal_share_cap": fed_share_cap,
                "claimed_federal_share": claimed_fed_share,
                "retainage_pct": retainage_pct,
                "days_since_invoice": days_since_invoice,
                "cost_variance_pct": cost_variance_pct,
                "narrative": narrative,
                "compliance_anomaly_flag": is_anomaly,
            }
        )

    return pd.DataFrame(rows)


def generate_gl_and_bank_reconciliation(seed: int = 42) -> pd.DataFrame:
    """Generate 600 multi-fund GL & bank statement lines with realistic exceptions."""
    rng = np.random.default_rng(seed + 2)
    funds = ["Operating_Fund_5070", "CIP_Grant_Fund_4910", "PFC_Fund_5075", "Debt_Service_Reserve_5080"]
    rows = []
    base_date = pd.Timestamp("2026-10-01")

    for idx in range(1, 301):
        fund = funds[idx % len(funds)]
        gl_dt = base_date + pd.Timedelta(days=int(rng.integers(0, 29)))
        amt = float(np.round(rng.uniform(1_250, 485_000), 2))
        ref = f"TXN-{idx:05d}"

        # Default: matched within 0-1 business days
        bank_dt = gl_dt + pd.Timedelta(days=int(rng.integers(0, 2)))
        bank_amt = amt
        bank_fund = fund
        true_status = "Matched"

        # Inject controlled exceptions (~12% total)
        if idx % 25 == 0:
            # Timing difference (deposit in transit at month-end)
            gl_dt = pd.Timestamp("2026-10-30")
            bank_dt = pd.NaT
            bank_amt = 0.0
            true_status = "Timing_Difference"
        elif idx % 25 == 7:
            # Unrecorded bank fee / treasury wire charge
            gl_dt = pd.NaT
            amt = 0.0
            bank_amt = float(np.round(rng.uniform(45, 450), 2))
            true_status = "Unrecorded_Bank_Fee"
        elif idx % 25 == 14:
            # Fund misclassification between Operating and PFC/CIP
            bank_fund = funds[(idx + 1) % len(funds)]
            true_status = "Fund_Misclassification"
        elif idx % 25 == 21:
            # Duplicate GL posting (GL has 2x bank)
            bank_amt = round(amt / 2.0, 2)
            true_status = "Duplicate_GL_Posting"

        rows.append(
            {
                "ref_id": ref,
                "gl_fund": fund,
                "bank_fund": bank_fund,
                "gl_post_date": "" if pd.isna(gl_dt) else gl_dt.strftime("%Y-%m-%d"),
                "bank_clear_date": "" if pd.isna(bank_dt) else bank_dt.strftime("%Y-%m-%d"),
                "gl_amount_usd": amt,
                "bank_amount_usd": bank_amt,
                "expected_category": true_status,
            }
        )

    return pd.DataFrame(rows)


def generate_debt_service_schedule() -> pd.DataFrame:
    """Generate FY2025-FY2034 Airport System Revenue Bond debt service schedule."""
    years = list(range(2025, 2035))
    # Calibrated to a mid-to-large hub expansion program (~$1.4B outstanding revenue bonds)
    senior_principal = [28_500_000, 30_100_000, 32_400_000, 34_800_000, 36_500_000,
                        38_200_000, 40_100_000, 42_000_000, 44_200_000, 46_500_000]
    senior_interest = [48_200_000, 46_800_000, 51_500_000, 50_100_000, 48_400_000,
                       46_600_000, 44_700_000, 42_700_000, 40_600_000, 38_400_000]
    sub_debt_service = [9_500_000, 9_500_000, 11_200_000, 11_200_000, 11_200_000,
                        10_800_000, 10_800_000, 10_500_000, 10_500_000, 10_200_000]

    rows = []
    for yr, p, intr, sub in zip(years, senior_principal, senior_interest, sub_debt_service):
        senior_total = p + intr
        rows.append(
            {
                "fiscal_year": yr,
                "senior_principal_usd": p,
                "senior_interest_usd": intr,
                "senior_debt_service_usd": senior_total,
                "subordinate_debt_service_usd": sub,
                "total_debt_service_usd": senior_total + sub,
                "senior_rate_covenant_min_dscr": 1.25,
            }
        )
    return pd.DataFrame(rows)


def generate_all_datasets(output_dir: Path, seed: int = 42) -> Dict[str, Path]:
    """Generate and persist all 4 synthetic datasets to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "enplanements_and_revenue": output_dir / "enplanements_and_revenue.csv",
        "cip_projects_and_faa_grants": output_dir / "cip_projects_and_faa_grants.csv",
        "gl_and_bank_reconciliation": output_dir / "gl_and_bank_reconciliation.csv",
        "debt_service_schedule": output_dir / "debt_service_schedule.csv",
    }
    generate_enplanements_and_revenue(seed).to_csv(files["enplanements_and_revenue"], index=False)
    generate_cip_and_faa_grants(seed).to_csv(files["cip_projects_and_faa_grants"], index=False)
    generate_gl_and_bank_reconciliation(seed).to_csv(files["gl_and_bank_reconciliation"], index=False)
    generate_debt_service_schedule().to_csv(files["debt_service_schedule"], index=False)
    logger.info("Generated 4 synthetic AUS aviation datasets in %s", output_dir)
    return files
