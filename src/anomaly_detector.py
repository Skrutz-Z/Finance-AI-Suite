"""General-ledger audit anomaly detection using Isolation Forest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

REQUIRED_GL_COLUMNS = (
    "Transaction_ID",
    "Amount",
    "Posting_Hour",
    "Account_Code",
    "Department",
    "Vendor",
)


@dataclass(frozen=True)
class AnomalyDetectionResult:
    """Container for scored journal entries, summary table, and visualization."""

    scored_entries: pd.DataFrame
    high_risk_summary: pd.DataFrame
    scatter_figure: go.Figure


class AuditAnomalyDetector:
    """Isolation Forest detector for suspicious general-ledger journal entries."""

    def __init__(
        self,
        contamination: float = 0.08,
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> None:
        """
        Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies in the dataset.
            random_state: Random seed for reproducibility.
            n_estimators: Number of isolation trees to fit.
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self._model: IsolationForest | None = None
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._feature_columns: list[str] = []
        self._fitted_entries: pd.DataFrame | None = None

    @staticmethod
    def _validate_gl_data(gl_df: pd.DataFrame) -> None:
        """Ensure the general-ledger DataFrame contains required columns."""
        missing = set(REQUIRED_GL_COLUMNS) - set(gl_df.columns)
        if missing:
            raise ValueError(f"gl_df is missing required columns: {sorted(missing)}")

    def _build_feature_matrix(self, gl_df: pd.DataFrame, fit_encoders: bool) -> pd.DataFrame:
        """Engineer numeric features for Isolation Forest training or scoring."""
        features = pd.DataFrame(index=gl_df.index)
        amounts = gl_df["Amount"].astype(float)
        features["amount"] = amounts
        features["abs_amount"] = amounts.abs()
        features["log_abs_amount"] = np.log1p(amounts.abs())
        features["posting_hour"] = gl_df["Posting_Hour"].astype(float)
        features["is_midnight_post"] = (gl_df["Posting_Hour"] <= 2).astype(float)
        features["is_round_amount"] = (amounts.abs() % 1000 == 0).astype(float)

        for column in ("Account_Code", "Department", "Vendor"):
            if fit_encoders:
                encoder = LabelEncoder()
                features[column] = encoder.fit_transform(gl_df[column].astype(str))
                self._label_encoders[column] = encoder
            else:
                encoder = self._label_encoders[column]
                values = gl_df[column].astype(str)
                known = set(encoder.classes_)
                safe_values = values.where(values.isin(known), encoder.classes_[0])
                features[column] = encoder.transform(safe_values)

        self._feature_columns = list(features.columns)
        return features

    def fit(self, gl_df: pd.DataFrame) -> None:
        """
        Fit an Isolation Forest model on general-ledger journal entries.

        Args:
            gl_df: DataFrame with Transaction_ID, Amount, Posting_Hour,
                Account_Code, Department, and Vendor columns.
        """
        self._validate_gl_data(gl_df)
        feature_matrix = self._build_feature_matrix(gl_df, fit_encoders=True)
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
        )
        self._model.fit(feature_matrix)
        self._fitted_entries = gl_df.copy()

    def _ensure_fitted(self) -> IsolationForest:
        """Validate that the model has been fit."""
        if self._model is None:
            raise RuntimeError("Call fit() before scoring anomalies.")
        return self._model

    def score_entries(self, gl_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Calculate anomaly scores and flags for journal entries.

        Lower anomaly scores indicate higher risk. Adds columns:
        anomaly_score, anomaly_flag, risk_level.

        Args:
            gl_df: Optional new entries to score. Uses fitted training data when omitted.

        Returns:
            DataFrame with original entry fields plus anomaly metrics.
        """
        model = self._ensure_fitted()
        source = gl_df.copy() if gl_df is not None else self._fitted_entries
        if source is None:
            raise RuntimeError("No data available to score. Provide gl_df or call fit() first.")
        self._validate_gl_data(source)

        fit_encoders = gl_df is not None and gl_df is not self._fitted_entries
        if fit_encoders and not self._label_encoders:
            raise RuntimeError("Label encoders unavailable. Call fit() before scoring new data.")

        feature_matrix = (
            self._build_feature_matrix(source, fit_encoders=False)
            if self._label_encoders
            else self._build_feature_matrix(source, fit_encoders=True)
        )

        raw_scores = model.score_samples(feature_matrix)
        predictions = model.predict(feature_matrix)

        scored = source.copy()
        scored["anomaly_score"] = np.round(raw_scores, 4)
        scored["anomaly_flag"] = predictions == -1
        scored["risk_level"] = np.where(
            scored["anomaly_flag"],
            "High",
            np.where(scored["anomaly_score"] < np.percentile(raw_scores, 25), "Medium", "Low"),
        )
        return scored.sort_values("anomaly_score", ascending=True).reset_index(drop=True)

    def get_high_risk_summary(
        self,
        scored_entries: pd.DataFrame,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Return a summary table of the highest-risk transactions.

        Args:
            scored_entries: Output from score_entries().
            top_n: Maximum number of rows to include in the summary.

        Returns:
            DataFrame sorted by ascending anomaly score (most anomalous first).
        """
        summary_columns = [
            "Transaction_ID",
            "Amount",
            "Posting_Hour",
            "Account_Code",
            "Department",
            "Vendor",
            "anomaly_score",
            "risk_level",
        ]
        high_risk = scored_entries[scored_entries["anomaly_flag"]].copy()
        if high_risk.empty:
            high_risk = scored_entries.nsmallest(top_n, "anomaly_score")
        else:
            high_risk = high_risk.nsmallest(top_n, "anomaly_score")
        return high_risk[summary_columns].reset_index(drop=True)

    def plot_anomaly_scatter(self, scored_entries: pd.DataFrame) -> go.Figure:
        """
        Build a Plotly scatter chart highlighting anomalous journal entries.

        Returns:
            plotly.graph_objects.Figure
        """
        normal = scored_entries[~scored_entries["anomaly_flag"]]
        outliers = scored_entries[scored_entries["anomaly_flag"]]

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=normal["Posting_Hour"],
                y=normal["Amount"],
                mode="markers",
                name="Normal Entries",
                marker={"color": "#1f77b4", "size": 8, "opacity": 0.65},
                text=normal["Transaction_ID"],
                hovertemplate=(
                    "ID: %{text}<br>Hour: %{x}<br>Amount: $%{y:,.2f}<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=outliers["Posting_Hour"],
                y=outliers["Amount"],
                mode="markers",
                name="Flagged Outliers",
                marker={
                    "color": "#d62728",
                    "size": 11,
                    "symbol": "x",
                    "line": {"width": 1.5, "color": "#8b0000"},
                },
                text=outliers["Transaction_ID"],
                hovertemplate=(
                    "ID: %{text}<br>Hour: %{x}<br>Amount: $%{y:,.2f}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            title="GL Journal Entry Anomaly Scatter (Posting Hour vs Amount)",
            xaxis_title="Posting Hour",
            yaxis_title="Amount ($)",
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )
        return figure

    def run_detection(
        self,
        gl_df: pd.DataFrame,
        top_n: int = 15,
    ) -> AnomalyDetectionResult:
        """
        Fit the model, score entries, and return summary data plus visualization.

        Args:
            gl_df: General-ledger journal entries.
            top_n: Number of high-risk rows in the summary table.

        Returns:
            AnomalyDetectionResult with scored data, summary table, and scatter figure.
        """
        self.fit(gl_df)
        scored = self.score_entries()
        summary = self.get_high_risk_summary(scored, top_n=top_n)
        figure = self.plot_anomaly_scatter(scored)
        return AnomalyDetectionResult(
            scored_entries=scored,
            high_risk_summary=summary,
            scatter_figure=figure,
        )
