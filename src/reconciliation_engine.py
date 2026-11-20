"""
Automated Multi-Fund General Ledger & Bank Reconciliation Engine.
Addresses JR105423: Lead general ledger and bank reconciliations, resolve account discrepancies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ReconciliationResult:
    reconciled_df: pd.DataFrame
    fund_summary_df: pd.DataFrame
    exception_summary_df: pd.DataFrame
    detection_accuracy_pct: float
    total_gl_usd: float
    total_bank_usd: float
    net_variance_usd: float
    reconciled_after_adjustments_gap_usd: float


def run_gl_bank_reconciliation(df: pd.DataFrame) -> ReconciliationResult:
    """Classify every GL/Bank pair, compute tie-outs, and verify 100% adjusting entry reconciliation."""
    rec = df.copy()
    rec["variance_usd"] = np.round(rec["gl_amount_usd"] - rec["bank_amount_usd"], 2)

    def _classify(row: pd.Series) -> str:
        gl_amt = float(row["gl_amount_usd"])
        bank_amt = float(row["bank_amount_usd"])
        gl_dt = "" if pd.isna(row["gl_post_date"]) else str(row["gl_post_date"]).strip()
        bank_dt = "" if pd.isna(row["bank_clear_date"]) else str(row["bank_clear_date"]).strip()
        gl_fund = "" if pd.isna(row["gl_fund"]) else str(row["gl_fund"]).strip()
        bank_fund = "" if pd.isna(row["bank_fund"]) else str(row["bank_fund"]).strip()

        if gl_amt > 0 and bank_amt == 0 and bank_dt == "":
            return "Timing_Difference"
        if gl_amt == 0 and bank_amt > 0 and gl_dt == "":
            return "Unrecorded_Bank_Fee"
        if abs(gl_amt - bank_amt) < 0.01 and gl_fund != bank_fund:
            return "Fund_Misclassification"
        if bank_amt > 0 and abs(gl_amt - 2.0 * bank_amt) < 0.05:
            return "Duplicate_GL_Posting"
        if abs(gl_amt - bank_amt) < 0.01 and gl_fund == bank_fund:
            return "Matched"
        return "Unresolved_Variance"

    rec["detected_category"] = rec.apply(_classify, axis=1)

    # Required adjusting entry (to bring GL book balance in line with adjusted bank balance)
    def _adj_entry(row: pd.Series) -> float:
        cat = row["detected_category"]
        if cat == "Unrecorded_Bank_Fee":
            return float(row["bank_amount_usd"])  # Record missing bank debit/credit in GL
        if cat == "Duplicate_GL_Posting":
            return -float(row["variance_usd"])    # Reverse duplicate GL entry
        return 0.0

    def _bank_transit_adj(row: pd.Series) -> float:
        if row["detected_category"] == "Timing_Difference":
            return float(row["gl_amount_usd"])    # Add deposit-in-transit to bank balance
        return 0.0

    rec["gl_adjusting_entry_usd"] = rec.apply(_adj_entry, axis=1)
    rec["bank_transit_adj_usd"] = rec.apply(_bank_transit_adj, axis=1)

    accuracy = float((rec["detected_category"] == rec["expected_category"]).mean() * 100.0)

    total_gl = round(float(rec["gl_amount_usd"].sum()), 2)
    total_bank = round(float(rec["bank_amount_usd"].sum()), 2)
    net_var = round(total_gl - total_bank, 2)

    adjusted_gl = round(float((rec["gl_amount_usd"] + rec["gl_adjusting_entry_usd"]).sum()), 2)
    adjusted_bank = round(float((rec["bank_amount_usd"] + rec["bank_transit_adj_usd"]).sum()), 2)
    post_adj_gap = round(adjusted_gl - adjusted_bank, 2)

    fund_summary = (
        rec.groupby("gl_fund")
        .agg(
            tx_count=("ref_id", "count"),
            gl_total_usd=("gl_amount_usd", "sum"),
            bank_total_usd=("bank_amount_usd", "sum"),
            raw_variance_usd=("variance_usd", "sum"),
            exception_count=("detected_category", lambda s: int((s != "Matched").sum())),
        )
        .reset_index()
    )

    exc_summary = (
        rec.groupby("detected_category")
        .agg(
            item_count=("ref_id", "count"),
            total_gl_usd=("gl_amount_usd", "sum"),
            total_bank_usd=("bank_amount_usd", "sum"),
            total_abs_variance_usd=("variance_usd", lambda s: float(np.abs(s).sum())),
        )
        .reset_index()
    )

    return ReconciliationResult(
        reconciled_df=rec,
        fund_summary_df=fund_summary,
        exception_summary_df=exc_summary,
        detection_accuracy_pct=round(accuracy, 2),
        total_gl_usd=total_gl,
        total_bank_usd=total_bank,
        net_variance_usd=net_var,
        reconciled_after_adjustments_gap_usd=post_adj_gap,
    )
