"""Unit tests for Prophet cash-flow model fitting and forecasting."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.data_generator import DataGeneratorConfig, generate_cash_flow_data
from src.forecaster import (
    MAX_HORIZON_DAYS,
    MIN_HORIZON_DAYS,
    CashFlowForecaster,
    ForecastScenario,
)


@pytest.fixture
def fitted_forecaster(cash_flow_df: pd.DataFrame) -> CashFlowForecaster:
    """Return a forecaster fitted on synthetic cash-flow data."""
    forecaster = CashFlowForecaster()
    forecaster.fit(cash_flow_df)
    return forecaster


class TestCashFlowForecaster:
    """Tests for CashFlowForecaster fit and forecast behavior."""

    def test_horizon_validation_bounds(self) -> None:
        with pytest.raises(ValueError):
            CashFlowForecaster._validate_horizon(MIN_HORIZON_DAYS - 1)
        with pytest.raises(ValueError):
            CashFlowForecaster._validate_horizon(MAX_HORIZON_DAYS + 1)
        assert CashFlowForecaster._validate_horizon(60) == 60

    def test_fit_missing_columns_raises(self, cash_flow_df: pd.DataFrame) -> None:
        forecaster = CashFlowForecaster()
        invalid_df = cash_flow_df.drop(columns=["net_cash_flow"])
        with pytest.raises(ValueError, match="missing required columns"):
            forecaster.fit(invalid_df)

    def test_fit_initializes_models(self, fitted_forecaster: CashFlowForecaster) -> None:
        assert fitted_forecaster._net_model is not None
        assert fitted_forecaster._revenue_model is not None
        assert fitted_forecaster._expense_model is not None
        assert fitted_forecaster._historical is not None

    def test_forecast_before_fit_raises(self) -> None:
        forecaster = CashFlowForecaster()
        with pytest.raises(RuntimeError, match="Call fit"):
            forecaster.forecast()

    def test_forecast_output_shape_and_columns(
        self,
        fitted_forecaster: CashFlowForecaster,
    ) -> None:
        horizon = 45
        net_df, revenue_df, expense_df = fitted_forecaster.forecast(horizon_days=horizon)
        assert len(net_df) == horizon
        assert len(revenue_df) == horizon
        assert len(expense_df) == horizon
        assert "net_cash_flow_forecast" in net_df.columns
        assert "net_cash_flow_lower" in net_df.columns
        assert "net_cash_flow_upper" in net_df.columns

    def test_confidence_intervals_are_ordered(
        self,
        fitted_forecaster: CashFlowForecaster,
    ) -> None:
        net_df, _, _ = fitted_forecaster.forecast(horizon_days=30)
        assert (net_df["net_cash_flow_lower"] <= net_df["net_cash_flow_forecast"]).all()
        assert (net_df["net_cash_flow_forecast"] <= net_df["net_cash_flow_upper"]).all()

    def test_scenario_adjustment_scales_forecast(
        self,
        fitted_forecaster: CashFlowForecaster,
    ) -> None:
        baseline, _, _ = fitted_forecaster.forecast(horizon_days=30)
        adjusted, _, _ = fitted_forecaster.forecast(
            horizon_days=30,
            scenario=ForecastScenario(net_cash_flow_adjustment_pct=10.0),
        )
        ratio = adjusted["net_cash_flow_forecast"].mean() / baseline["net_cash_flow_forecast"].mean()
        assert ratio == pytest.approx(1.10, rel=1e-6)

    def test_run_forecast_returns_plotly_figures(self, cash_flow_df: pd.DataFrame) -> None:
        forecaster = CashFlowForecaster()
        result = forecaster.run_forecast(cash_flow_df, horizon_days=30)
        assert isinstance(result.net_cash_flow_figure, go.Figure)
        assert isinstance(result.components_figure, go.Figure)
        assert len(result.net_cash_flow_figure.data) >= 2

    def test_smaller_dataset_still_fits(self) -> None:
        config = DataGeneratorConfig(cash_flow_days=90, random_seed=1)
        df = generate_cash_flow_data(config)
        forecaster = CashFlowForecaster()
        forecaster.fit(df)
        net_df, _, _ = forecaster.forecast(horizon_days=30)
        assert len(net_df) == 30
