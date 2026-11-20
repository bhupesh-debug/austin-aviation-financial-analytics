"""
ML Forecasting Engine for Municipal Airport Operating Revenue & Enplanements.
Addresses JR105423: Operating/CIP Budgeting, Revenue & Economic Forecasting, Variance Analysis.

Enforces strict chronological train/test splits (zero future data leakage):
- Train: FY2019–FY2024 (first 72 months)
- Test:  FY2025–FY2026 (final 24 months)
Benchmarks: Seasonal Naive -> Ridge Regression -> Random Forest -> Gradient Boosting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastResult:
    metrics_df: pd.DataFrame
    predictions_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    best_model_name: str
    fy2027_projected_revenue_usd: float
    fy2027_projected_expense_usd: float
    fy2027_projected_noi_usd: float


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-free lag, rolling, seasonal, and macro features."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    # Cyclical seasonality encoding
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)

    # Strictly lagged target & driver features (no contemporaneous leakage)
    for lag in [1, 3, 12]:
        data[f"rev_lag_{lag}"] = data["total_operating_revenue"].shift(lag)
        data[f"enp_lag_{lag}"] = data["enplanements"].shift(lag)

    data["rev_roll_3_mean"] = data["total_operating_revenue"].shift(1).rolling(3).mean()
    data["rev_roll_12_mean"] = data["total_operating_revenue"].shift(1).rolling(12).mean()
    data["emp_lag_1"] = data["austin_metro_emp_idx"].shift(1)
    data["fuel_lag_1"] = data["jet_fuel_price_gal"].shift(1)
    data["cpi_lag_1"] = data["cpi_index"].shift(1)
    data["yoy_enp_growth_lag1"] = (data["enp_lag_1"] - data["enp_lag_12"]) / data["enp_lag_12"]

    return data.dropna().reset_index(drop=True)


def _calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE_USD": round(mae, 2), "RMSE_USD": round(rmse, 2), "MAPE_Pct": round(mape, 2), "WAPE_Pct": round(wape, 2), "R2": round(r2, 4)}


def train_and_evaluate_forecasters(df: pd.DataFrame, seed: int = 42) -> ForecastResult:
    """Run chronological benchmark of Baseline, Ridge, Random Forest, and Gradient Boosting."""
    feat_df = build_features(df)

    feature_cols = [
        "month_sin", "month_cos", "rev_lag_1", "rev_lag_3", "rev_lag_12",
        "enp_lag_1", "enp_lag_3", "enp_lag_12", "rev_roll_3_mean", "rev_roll_12_mean",
        "emp_lag_1", "fuel_lag_1", "cpi_lag_1", "yoy_enp_growth_lag1",
    ]
    target_col = "total_operating_revenue"

    # Strict chronological split: last 24 months (FY2025-FY2026) as out-of-sample test
    test_size = 24
    train_df = feat_df.iloc[:-test_size].copy()
    test_df = feat_df.iloc[-test_size:].copy()

    X_train, y_train = train_df[feature_cols], train_df[target_col].values
    X_test, y_test = test_df[feature_cols], test_df[target_col].values

    models: Dict[str, Any] = {
        "Ridge_Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=5.0))]),
        "Random_Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=seed),
        "Gradient_Boosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.05, max_depth=3, random_state=seed),
    }

    metrics_rows = []
    pred_df = test_df[["date", "fiscal_year", "month", target_col, "budget_revenue"]].copy()

    # 1. Baseline: Seasonal Naive (t - 12)
    naive_pred = test_df["rev_lag_12"].values
    pred_df["Seasonal_Naive"] = np.round(naive_pred, 2)
    m_naive = _calc_metrics(y_test, naive_pred)
    metrics_rows.append({"Model": "Seasonal_Naive_Baseline", **m_naive})

    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        pred_df[name] = np.round(preds, 2)
        m = _calc_metrics(y_test, preds)
        metrics_rows.append({"Model": name, **m})
        fitted_models[name] = model

    metrics_df = pd.DataFrame(metrics_rows).sort_values("WAPE_Pct").reset_index(drop=True)
    best_model_name = str(metrics_df.iloc[0]["Model"])

    # Feature importance from Gradient Boosting (tree-based interpretability)
    gb_model = fitted_models["Gradient_Boosting"]
    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": gb_model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    # FY2027 12-month budget projection (growth-adjusted from FY2026 actuals)
    fy26 = df[df["fiscal_year"] == 2026]
    fy25 = df[df["fiscal_year"] == 2025]
    rev_growth = float((fy26["total_operating_revenue"].sum() / fy25["total_operating_revenue"].sum()) - 1.0)
    exp_growth = float((fy26["total_operating_expense"].sum() / fy25["total_operating_expense"].sum()) - 1.0)

    # Conservative municipal budget growth cap
    proj_rev_growth = float(np.clip(rev_growth, 0.02, 0.065))
    proj_exp_growth = float(np.clip(exp_growth, 0.025, 0.060))

    fy27_rev = round(float(fy26["total_operating_revenue"].sum() * (1.0 + proj_rev_growth)), 2)
    fy27_exp = round(float(fy26["total_operating_expense"].sum() * (1.0 + proj_exp_growth)), 2)
    fy27_noi = round(fy27_rev - fy27_exp, 2)

    return ForecastResult(
        metrics_df=metrics_df,
        predictions_df=pred_df,
        feature_importance_df=fi_df,
        best_model_name=best_model_name,
        fy2027_projected_revenue_usd=fy27_rev,
        fy2027_projected_expense_usd=fy27_exp,
        fy2027_projected_noi_usd=fy27_noi,
    )
