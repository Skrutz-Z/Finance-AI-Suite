"""Unit tests for GL audit anomaly detection."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.anomaly_detector import AuditAnomalyDetector


class TestAuditAnomalyDetector:
    """Tests for Isolation Forest anomaly detection."""

    def test_fit_missing_columns_raises(self, gl_df: pd.DataFrame) -> None:
        detector = AuditAnomalyDetector()
        invalid_df = gl_df.drop(columns=["Vendor"])
        with pytest.raises(ValueError, match="missing required columns"):
            detector.fit(invalid_df)

    def test_fit_and_score_entries(self, gl_df: pd.DataFrame) -> None:
        detector = AuditAnomalyDetector(contamination=0.10, random_state=42)
        detector.fit(gl_df)
        scored = detector.score_entries()
        assert len(scored) == len(gl_df)
        assert {"anomaly_score", "anomaly_flag", "risk_level"}.issubset(scored.columns)

    def test_score_before_fit_raises(self, gl_df: pd.DataFrame) -> None:
        detector = AuditAnomalyDetector()
        with pytest.raises(RuntimeError, match="Call fit"):
            detector.score_entries(gl_df)

    def test_flagged_count_aligns_with_contamination(self, gl_df: pd.DataFrame) -> None:
        contamination = 0.10
        detector = AuditAnomalyDetector(contamination=contamination, random_state=42)
        detector.fit(gl_df)
        scored = detector.score_entries()
        flagged = scored["anomaly_flag"].sum()
        expected = int(contamination * len(gl_df))
        assert flagged == expected

    def test_risk_levels_are_valid(self, gl_df: pd.DataFrame) -> None:
        detector = AuditAnomalyDetector()
        result = detector.run_detection(gl_df)
        assert result.scored_entries["risk_level"].isin(["Low", "Medium", "High"]).all()

    def test_high_risk_summary_shape(self, gl_df: pd.DataFrame) -> None:
        detector = AuditAnomalyDetector(contamination=0.10)
        result = detector.run_detection(gl_df, top_n=10)
        assert len(result.high_risk_summary) <= 10
        assert "Transaction_ID" in result.high_risk_summary.columns
        assert "anomaly_score" in result.high_risk_summary.columns

    def test_scatter_figure_is_plotly(self, gl_df: pd.DataFrame) -> None:
        detector = AuditAnomalyDetector()
        result = detector.run_detection(gl_df)
        assert isinstance(result.scatter_figure, go.Figure)
        assert len(result.scatter_figure.data) == 2

    def test_anomaly_scores_are_numeric(self, gl_df: pd.DataFrame) -> None:
        detector = AuditAnomalyDetector()
        detector.fit(gl_df)
        scored = detector.score_entries()
        assert pd.api.types.is_numeric_dtype(scored["anomaly_score"])

    def test_flagged_entries_have_lowest_scores(self, gl_df: pd.DataFrame) -> None:
        detector = AuditAnomalyDetector(contamination=0.10, random_state=42)
        scored = detector.run_detection(gl_df).scored_entries
        flagged_scores = scored.loc[scored["anomaly_flag"], "anomaly_score"]
        normal_scores = scored.loc[~scored["anomaly_flag"], "anomaly_score"]
        assert flagged_scores.max() <= normal_scores.min()
