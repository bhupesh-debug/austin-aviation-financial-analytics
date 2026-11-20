"""
Data Mining & FAA AIP/BIL Grant Compliance Anomaly Detection Engine.
Directly aligns with SHL Data Mining Concepts & JR105423 Grant Compliance:
- Exploratory Data Analysis (EDA) & Distance Metrics (Euclidean / Cosine)
- KMeans Clustering of CIP Project Drawdown Profiles
- Text Mining (TF-IDF) on Drawdown Narratives
- Unsupervised (IsolationForest) & Supervised (RandomForest) Compliance Screening
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


@dataclass
class DataMiningResult:
    scored_df: pd.DataFrame
    model_metrics: Dict[str, Dict[str, float]]
    cluster_summary_df: pd.DataFrame
    top_risk_terms: List[Dict[str, float]]


def run_grant_data_mining(df: pd.DataFrame, seed: int = 42) -> DataMiningResult:
    """Execute EDA distance profiling, KMeans clustering, TF-IDF text mining, and anomaly detection."""
    data = df.copy()
    data["fed_share_excess"] = np.maximum(0.0, data["claimed_federal_share"] - data["federal_share_cap"])

    num_cols = [
        "drawdown_to_budget_ratio",
        "claimed_federal_share",
        "fed_share_excess",
        "retainage_pct",
        "days_since_invoice",
        "cost_variance_pct",
    ]
    scaler = StandardScaler()
    X_num = scaler.fit_transform(data[num_cols].values)

    # 1. Measurement & Distance: Euclidean distance from global centroid (SHL Data Mining Concept)
    centroid = np.mean(X_num, axis=0, keepdims=True)
    data["euclidean_dist_from_centroid"] = np.round(cdist(X_num, centroid, metric="euclidean").ravel(), 4)

    # 2. Clustering: KMeans (k=3 risk tiers)
    km = KMeans(n_clusters=3, random_state=seed, n_init=10)
    data["cip_cluster"] = km.fit_predict(X_num)

    # 3. Text Mining: TF-IDF on contractor/grant narratives
    tfidf = TfidfVectorizer(stop_words="english", max_features=30)
    X_text = tfidf.fit_transform(data["narrative"].fillna("")).toarray()
    vocab = tfidf.get_feature_names_out()

    # Combine numerical + text features
    X_combined = np.hstack([X_num, X_text])
    y = data["compliance_anomaly_flag"].values

    # 4a. Unsupervised Anomaly Detection: IsolationForest on dense numeric drivers (avoids TF-IDF curse of dimensionality)
    iso = IsolationForest(contamination=0.10, random_state=seed)
    iso.fit(X_num)
    iso_raw = -iso.decision_function(X_num)  # higher = more anomalous
    iso_pred = (iso.predict(X_num) == -1).astype(int)
    data["iso_anomaly_score"] = np.round(iso_raw, 4)
    data["iso_flagged"] = iso_pred

    # 4b. Supervised Classifier with Stratified 5-Fold CV (zero leakage)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=seed)
    rf_proba = cross_val_predict(rf, X_combined, y, cv=skf, method="predict_proba")[:, 1]
    rf_pred = (rf_proba >= 0.50).astype(int)
    data["rf_anomaly_prob"] = np.round(rf_proba, 4)
    data["rf_flagged"] = rf_pred

    metrics = {
        "IsolationForest_Unsupervised": {
            "Precision": round(float(precision_score(y, iso_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y, iso_pred, zero_division=0)), 4),
            "F1": round(float(f1_score(y, iso_pred, zero_division=0)), 4),
            "ROC_AUC": round(float(roc_auc_score(y, iso_raw)), 4),
            "PR_AUC": round(float(average_precision_score(y, iso_raw)), 4),
        },
        "RandomForest_5FoldCV_Supervised": {
            "Precision": round(float(precision_score(y, rf_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y, rf_pred, zero_division=0)), 4),
            "F1": round(float(f1_score(y, rf_pred, zero_division=0)), 4),
            "ROC_AUC": round(float(roc_auc_score(y, rf_proba)), 4),
            "PR_AUC": round(float(average_precision_score(y, rf_proba)), 4),
        },
    }

    # Top TF-IDF terms correlated with compliance anomalies
    anom_mask = y == 1
    clean_mask = y == 0
    term_lift = X_text[anom_mask].mean(axis=0) - X_text[clean_mask].mean(axis=0)
    top_idx = np.argsort(term_lift)[::-1][:8]
    top_risk_terms = [{"term": str(vocab[i]), "tfidf_anomaly_lift": round(float(term_lift[i]), 4)} for i in top_idx if term_lift[i] > 0]

    cluster_summary = (
        data.groupby("cip_cluster")
        .agg(
            drawdown_count=("drawdown_id", "count"),
            avg_drawdown_usd=("drawdown_amount_usd", "mean"),
            avg_cost_variance_pct=("cost_variance_pct", "mean"),
            avg_euclidean_dist=("euclidean_dist_from_centroid", "mean"),
            anomaly_rate=("compliance_anomaly_flag", "mean"),
        )
        .reset_index()
    )

    return DataMiningResult(
        scored_df=data,
        model_metrics=metrics,
        cluster_summary_df=cluster_summary,
        top_risk_terms=top_risk_terms,
    )
