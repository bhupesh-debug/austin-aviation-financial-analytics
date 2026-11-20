"""
Excel 2019 Power User Financial Model Builder (openpyxl).
Demonstrates SHL Microsoft Excel 2019 Power User competencies:
- Dynamic cross-sheet formulas (SUMIFS, INDEX/MATCH, IFERROR, AVERAGE, IF)
- Institutional formatting, accounting number formats, frozen panes, conditional formatting
- 5 Tabs: Executive_Dashboard, Operating_Budget_Variance, CIP_FAA_Grants, Debt_DSCR_Model, GL_Bank_Reconciliation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import openpyxl
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd


NAVY_FILL = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
SUBHEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
ALERT_RED = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
OK_GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")

HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
BOLD_FONT = Font(name="Calibri", size=10, bold=True, color="1B365D")
BODY_FONT = Font(name="Calibri", size=10, color="000000")
TITLE_FONT = Font(name="Calibri", size=14, bold=True, color="1B365D")

THIN_BORDER = Border(
    left=Side(style="thin", color="BFBFBF"),
    right=Side(style="thin", color="BFBFBF"),
    top=Side(style="thin", color="BFBFBF"),
    bottom=Side(style="thin", color="BFBFBF"),
)


def _style_header_row(ws, row_idx: int, max_col: int) -> None:
    for col in range(1, max_col + 1):
        c = ws.cell(row=row_idx, column=col)
        c.fill = NAVY_FILL
        c.font = HEADER_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = THIN_BORDER
    ws.row_dimensions[row_idx].height = 24


def _autofit(ws, min_width: int = 13, max_width: int = 34) -> None:
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val = "" if cell.value is None else str(cell.value)
            if val.startswith("="):
                val = "123,456,789.00"
            max_len = max(max_len, len(val))
        ws.column_dimensions[col_letter].width = max(min_width, min(max_width, max_len + 3))


def build_excel_power_user_workbook(
    rev_df: pd.DataFrame,
    cip_df: pd.DataFrame,
    debt_df: pd.DataFrame,
    rec_df: pd.DataFrame,
    mc_stats: Dict[str, float],
    output_path: Path,
) -> Path:
    """Create a 5-tab dynamic Excel 2019 Power User financial model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()

    # ------------------------------------------------------------------
    # TAB 1: Executive_Dashboard (Cross-sheet KPI formulas)
    # ------------------------------------------------------------------
    ws_dash = wb.active
    ws_dash.title = "Executive_Dashboard"
    ws_dash.views.sheetView[0].showGridLines = True

    ws_dash["A1"] = "AUSTIN-BERGSTROM INTERNATIONAL AIRPORT (AUS) — FINANCIAL ANALYTICS MODEL [SYN]"
    ws_dash["A1"].font = TITLE_FONT
    ws_dash["A2"] = "Prepared for City of Austin Dept. of Aviation (JR105423) | Dynamic Excel 2019 Power User Workbook"
    ws_dash["A2"].font = Font(name="Calibri", size=10, italic=True, color="404040")

    dash_headers = ["KPI / Executive Control Metric", "Formula-Driven Value", "Benchmark / Covenant", "Status"]
    for idx, h in enumerate(dash_headers, start=1):
        ws_dash.cell(row=4, column=idx, value=h)
    _style_header_row(ws_dash, 4, 4)

    dash_rows = [
        ("FY2026 Total Operating Revenue ($)", "=SUMIFS(Operating_Budget_Variance!E:E,Operating_Budget_Variance!B:B,2026)", "$250M+ Hub Scale", '=IF(B5>0,"ACTIVE","CHECK")'),
        ("FY2026 Budget Revenue Variance ($)", "=SUMIFS(Operating_Budget_Variance!G:G,Operating_Budget_Variance!B:B,2026)", "Target >= $0", '=IF(B6>=0,"FAVORABLE","UNFAVORABLE")'),
        ("FY2026 Total Operating Expense ($)", "=SUMIFS(Operating_Budget_Variance!H:H,Operating_Budget_Variance!B:B,2026)", "Controlled O&M", '=IF(B7>0,"TRACKED","CHECK")'),
        ("FY2026 Net Operating Income ($)", "=B5-B7", "Positive Enterprise NOI", '=IF(B8>0,"SURPLUS","DEFICIT")'),
        ("FY2026 Operating Margin (%)", "=IFERROR(B8/B5,0)", "Target >= 25.0%", '=IF(B9>=0.25,"STRONG","WATCH")'),
        ("FY2026 Senior Bond DSCR (x)", "=INDEX(Debt_DSCR_Model!G:G,MATCH(2026,Debt_DSCR_Model!A:A,0))", "Min Covenant >= 1.25x", '=IF(B10>=1.25,"COMPLIANT","BREACH")'),
        ("FY2027 Monte Carlo Median DSCR (x)", mc_stats.get("median_dscr_x", 0.0), "10,000-Run Simulation", '=IF(B11>=1.25,"COMPLIANT","WATCH")'),
        ("FY2027 Covenant Compliance Prob (%)", mc_stats.get("prob_meet_covenant_pct", 0.0) / 100.0, "Target >= 90.0%", '=IF(B12>=0.9,"ROBUST","REVIEW")'),
        ("CIP / FAA Grant Flagged Drawdowns (#)", '=COUNTIF(CIP_FAA_Grants!L:L,1)', "Compliance Screening", '=IF(B13>0,"AUDIT QUEUE","CLEAN")'),
        ("GL vs Bank Reconciliation Exceptions (#)", '=COUNTIF(GL_Bank_Reconciliation!I:I,"<>Matched")', "100% Tie-Out Required", '=IF(B14>0,"ACTION REQ","RECONCILED")'),
    ]

    for r_offset, (label, formula, bench, status) in enumerate(dash_rows, start=5):
        ws_dash.cell(row=r_offset, column=1, value=label).font = BOLD_FONT
        c_val = ws_dash.cell(row=r_offset, column=2, value=formula)
        c_val.font = BODY_FONT
        if "($)" in label:
            c_val.number_format = "$#,##0"
        elif "(%)" in label:
            c_val.number_format = "0.0%"
        elif "(x)" in label:
            c_val.number_format = "0.00\"x\""
        else:
            c_val.number_format = "#,##0"
        ws_dash.cell(row=r_offset, column=3, value=bench).font = BODY_FONT
        ws_dash.cell(row=r_offset, column=4, value=status).font = BOLD_FONT
        for col in range(1, 5):
            ws_dash.cell(row=r_offset, column=col).border = THIN_BORDER

    ws_dash.conditional_formatting.add(
        "D5:D14", CellIsRule(operator="equal", formula=['"COMPLIANT"'], fill=OK_GREEN)
    )
    ws_dash.conditional_formatting.add(
        "D5:D14", CellIsRule(operator="equal", formula=['"BREACH"'], fill=ALERT_RED)
    )
    ws_dash.freeze_panes = "A5"
    _autofit(ws_dash)

    # ------------------------------------------------------------------
    # TAB 2: Operating_Budget_Variance (FY2025-FY2026 Monthly Detail)
    # ------------------------------------------------------------------
    ws_op = wb.create_sheet("Operating_Budget_Variance")
    op_headers = [
        "Date", "Fiscal Year", "Enplanements", "Budget Revenue ($)", "Actual Revenue ($)",
        "Rev Variance (%)", "Rev Variance ($)", "Actual Expense ($)", "Budget Expense ($)",
        "Exp Variance ($)", "Net Operating Income ($)", "Operating Margin (%)"
    ]
    for col, h in enumerate(op_headers, start=1):
        ws_op.cell(row=1, column=col, value=h)
    _style_header_row(ws_op, 1, len(op_headers))

    sub_rev = rev_df[rev_df["fiscal_year"].isin([2025, 2026])].reset_index(drop=True)
    for idx, row in sub_rev.iterrows():
        r = idx + 2
        ws_op.cell(row=r, column=1, value=str(row["date"]))
        ws_op.cell(row=r, column=2, value=int(row["fiscal_year"]))
        ws_op.cell(row=r, column=3, value=int(row["enplanements"])).number_format = "#,##0"
        ws_op.cell(row=r, column=4, value=float(row["budget_revenue"])).number_format = "$#,##0"
        ws_op.cell(row=r, column=5, value=float(row["total_operating_revenue"])).number_format = "$#,##0"
        ws_op.cell(row=r, column=6, value=f"=IFERROR((E{r}-D{r})/D{r},0)").number_format = "0.0%"
        ws_op.cell(row=r, column=7, value=f"=E{r}-D{r}").number_format = "$#,##0"
        ws_op.cell(row=r, column=8, value=float(row["total_operating_expense"])).number_format = "$#,##0"
        ws_op.cell(row=r, column=9, value=float(row["budget_expense"])).number_format = "$#,##0"
        ws_op.cell(row=r, column=10, value=f"=I{r}-H{r}").number_format = "$#,##0"
        ws_op.cell(row=r, column=11, value=f"=E{r}-H{r}").number_format = "$#,##0"
        ws_op.cell(row=r, column=12, value=f"=IFERROR(K{r}/E{r},0)").number_format = "0.0%"

    ws_op.freeze_panes = "A2"
    ws_op.auto_filter.ref = f"A1:L{len(sub_rev)+1}"
    _autofit(ws_op)

    # ------------------------------------------------------------------
    # TAB 3: CIP_FAA_Grants (First 80 Drawdowns + Excess Formula)
    # ------------------------------------------------------------------
    ws_cip = wb.create_sheet("CIP_FAA_Grants")
    cip_headers = [
        "Drawdown ID", "Program", "Grant Type", "Project Budget ($)", "Drawdown ($)",
        "Drawdown %", "Fed Share Cap", "Claimed Fed Share", "Fed Share Excess",
        "Cost Variance %", "ISO Anomaly Score", "Flagged (1=Yes)"
    ]
    for col, h in enumerate(cip_headers, start=1):
        ws_cip.cell(row=1, column=col, value=h)
    _style_header_row(ws_cip, 1, len(cip_headers))

    for idx, row in cip_df.head(80).iterrows():
        r = idx + 2
        ws_cip.cell(row=r, column=1, value=str(row["drawdown_id"]))
        ws_cip.cell(row=r, column=2, value=str(row["program"]))
        ws_cip.cell(row=r, column=3, value=str(row["grant_type"]))
        ws_cip.cell(row=r, column=4, value=float(row["project_budget_usd"])).number_format = "$#,##0"
        ws_cip.cell(row=r, column=5, value=float(row["drawdown_amount_usd"])).number_format = "$#,##0"
        ws_cip.cell(row=r, column=6, value=f"=IFERROR(E{r}/D{r},0)").number_format = "0.0%"
        ws_cip.cell(row=r, column=7, value=float(row["federal_share_cap"])).number_format = "0.0%"
        ws_cip.cell(row=r, column=8, value=float(row["claimed_federal_share"])).number_format = "0.0%"
        ws_cip.cell(row=r, column=9, value=f"=MAX(0,H{r}-G{r})").number_format = "0.0%"
        ws_cip.cell(row=r, column=10, value=float(row["cost_variance_pct"])).number_format = "0.0%"
        ws_cip.cell(row=r, column=11, value=float(row.get("iso_anomaly_score", 0.0))).number_format = "0.000"
        ws_cip.cell(row=r, column=12, value=int(row.get("iso_flagged", 0)))

    ws_cip.conditional_formatting.add("L2:L81", CellIsRule(operator="equal", formula=["1"], fill=ALERT_RED))
    ws_cip.freeze_panes = "A2"
    ws_cip.auto_filter.ref = "A1:L81"
    _autofit(ws_cip)

    # ------------------------------------------------------------------
    # TAB 4: Debt_DSCR_Model
    # ------------------------------------------------------------------
    ws_debt = wb.create_sheet("Debt_DSCR_Model")
    debt_headers = [
        "Fiscal Year", "Senior Principal ($)", "Senior Interest ($)", "Total Senior DS ($)",
        "Subordinate DS ($)", "Net Rev Available ($)", "Senior DSCR (x)", "Covenant Min (x)", "Covenant Check"
    ]
    for col, h in enumerate(debt_headers, start=1):
        ws_debt.cell(row=1, column=col, value=h)
    _style_header_row(ws_debt, 1, len(debt_headers))

    fy26_noi = float(rev_df[rev_df["fiscal_year"] == 2026]["net_operating_income"].sum())
    fy26_pfc = float(rev_df[rev_df["fiscal_year"] == 2026]["pfc_revenue"].sum())
    base_net_avail = fy26_noi + 0.25 * fy26_pfc

    for idx, row in debt_df.iterrows():
        r = idx + 2
        yr = int(row["fiscal_year"])
        # Scale Net Rev Available by 3% annual growth from FY2026 anchor
        est_net = base_net_avail * ((1.03) ** (yr - 2026))
        ws_debt.cell(row=r, column=1, value=yr)
        ws_debt.cell(row=r, column=2, value=float(row["senior_principal_usd"])).number_format = "$#,##0"
        ws_debt.cell(row=r, column=3, value=float(row["senior_interest_usd"])).number_format = "$#,##0"
        ws_debt.cell(row=r, column=4, value=f"=B{r}+C{r}").number_format = "$#,##0"
        ws_debt.cell(row=r, column=5, value=float(row["subordinate_debt_service_usd"])).number_format = "$#,##0"
        ws_debt.cell(row=r, column=6, value=round(est_net, 2)).number_format = "$#,##0"
        ws_debt.cell(row=r, column=7, value=f"=IFERROR(F{r}/D{r},0)").number_format = "0.00\"x\""
        ws_debt.cell(row=r, column=8, value=float(row["senior_rate_covenant_min_dscr"])).number_format = "0.00\"x\""
        ws_debt.cell(row=r, column=9, value=f'=IF(G{r}>=H{r},"PASS","BREACH")')

    ws_debt.freeze_panes = "A2"
    _autofit(ws_debt)

    # ------------------------------------------------------------------
    # TAB 5: GL_Bank_Reconciliation (First 100 Lines + Tie-Out Formula)
    # ------------------------------------------------------------------
    ws_rec = wb.create_sheet("GL_Bank_Reconciliation")
    rec_headers = [
        "Ref ID", "GL Fund", "Bank Fund", "GL Post Date", "Bank Clear Date",
        "GL Amount ($)", "Bank Amount ($)", "Variance ($)", "Detected Category", "GL Adj Entry ($)"
    ]
    for col, h in enumerate(rec_headers, start=1):
        ws_rec.cell(row=1, column=col, value=h)
    _style_header_row(ws_rec, 1, len(rec_headers))

    for idx, row in rec_df.head(100).iterrows():
        r = idx + 2
        ws_rec.cell(row=r, column=1, value=str(row["ref_id"]))
        ws_rec.cell(row=r, column=2, value=str(row["gl_fund"]))
        ws_rec.cell(row=r, column=3, value=str(row["bank_fund"]))
        ws_rec.cell(row=r, column=4, value=str(row["gl_post_date"]))
        ws_rec.cell(row=r, column=5, value=str(row["bank_clear_date"]))
        ws_rec.cell(row=r, column=6, value=float(row["gl_amount_usd"])).number_format = "$#,##0.00"
        ws_rec.cell(row=r, column=7, value=float(row["bank_amount_usd"])).number_format = "$#,##0.00"
        ws_rec.cell(row=r, column=8, value=f"=F{r}-G{r}").number_format = "$#,##0.00"
        ws_rec.cell(row=r, column=9, value=str(row["detected_category"]))
        ws_rec.cell(row=r, column=10, value=float(row["gl_adjusting_entry_usd"])).number_format = "$#,##0.00"

    ws_rec.freeze_panes = "A2"
    ws_rec.auto_filter.ref = "A1:J101"
    _autofit(ws_rec)

    wb.save(output_path)
    return output_path
