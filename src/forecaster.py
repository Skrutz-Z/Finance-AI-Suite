"""Cash-flow forecasting using Facebook Prophet."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

MIN_HORIZON_DAYS = 30
MAX_HORIZON_DAYS = 90
CONFIDENCE_INTERVAL = 0.95


@dataclass(frozen=True)
class ForecastScenario:
    """Scenario adjustments applied to forecast outputs."""

    revenue_adjustment_pct: float = 0.0
    expense_adjustment_pct: float = 0.0
    net_cash_flow_adjustment_pct: float = 0.0


@dataclass(frozen=True)
class ForecastResult:
    """Container for Prophet forecast outputs and visualization."""

    net_cash_flow: pd.DataFrame
    revenue: pd.DataFrame
    expenses: pd.DataFrame
    net_cash_flow_figure: go.Figure
    components_figure: go.Figure


class CashFlowForecaster:
    """Prophet-based forecaster for daily cash-flow time series."""

    def __init__(self, random_seed: int = 42) -> None:
        """Initialize forecaster models and runtime state."""
        self.random_seed = random_seed
        self._historical: pd.DataFrame | None = None
        self._net_model: Prophet | None = None
        self._revenue_model: Prophet | None = None
        self._expense_model: Prophet | None = None

    @staticmethod
    def _validate_horizon(horizon_days: int) -> int:
        """Ensure forecast horizon is within supported bounds."""
        if not MIN_HORIZON_DAYS <= horizon_days <= MAX_HORIZON_DAYS:
            raise ValueError(
                f"horizon_days must be between {MIN_HORIZON_DAYS} and {MAX_HORIZON_DAYS}, "
                f"got {horizon_days}."
            )
        return horizon_days

    @staticmethod
    def _to_prophet_frame(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Convert project cash-flow schema to Prophet ds/y columns."""
        prophet_df = df[[date_col, value_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        return prophet_df.sort_values("ds").reset_index(drop=True)

    @staticmethod
    def _build_prophet_model() -> Prophet:
        """Create a Prophet model configured for 95% confidence intervals."""
        return Prophet(
            interval_width=CONFIDENCE_INTERVAL,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )

    def fit(self, cash_flow_df: pd.DataFrame) -> None:
        """
        Fit Prophet models for net cash flow, revenue, and expenses.

        Args:
            cash_flow_df: DataFrame with columns date, revenue, expenses, net_cash_flow.
        """
        required = {"date", "revenue", "expenses", "net_cash_flow"}
        missing = required - set(cash_flow_df.columns)
        if missing:
            raise ValueError(f"cash_flow_df is missing required columns: {sorted(missing)}")

        self._historical = cash_flow_df.copy()
        self._historical["date"] = pd.to_datetime(self._historical["date"])

        self._net_model = self._build_prophet_model()
        self._revenue_model = self._build_prophet_model()
        self._expense_model = self._build_prophet_model()

        self._net_model.fit(self._to_prophet_frame(cash_flow_df, "date", "net_cash_flow"))
        self._revenue_model.fit(self._to_prophet_frame(cash_flow_df, "date", "revenue"))
        self._expense_model.fit(self._to_prophet_frame(cash_flow_df, "date", "expenses"))

    def _ensure_fitted(self) -> tuple[Prophet, Prophet, Prophet, pd.DataFrame]:
        """Validate that fit has been called and return fitted artifacts."""
        if (
            self._historical is None
            or self._net_model is None
            or self._revenue_model is None
            or self._expense_model is None
        ):
            raise RuntimeError("Call fit() before generating forecasts.")
        return self._net_model, self._revenue_model, self._expense_model, self._historical

    def _forecast_series(
        self,
        model: Prophet,
        horizon_days: int,
        adjustment_pct: float,
        value_name: str,
    ) -> pd.DataFrame:
        """Generate and scenario-adjust a single Prophet forecast series."""
        future = model.make_future_dataframe(periods=horizon_days, freq="D")
        raw = model.predict(future)
        result = raw.tail(horizon_days)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        multiplier = 1.0 + (adjustment_pct / 100.0)
        result[f"{value_name}_forecast"] = result["yhat"] * multiplier
        result[f"{value_name}_lower"] = result["yhat_lower"] * multiplier
        result[f"{value_name}_upper"] = result["yhat_upper"] * multiplier
        return result[["ds", f"{value_name}_forecast", f"{value_name}_lower", f"{value_name}_upper"]]

    def forecast(
        self,
        horizon_days: int = 60,
        scenario: ForecastScenario | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate scenario-adjusted forecasts for net cash flow, revenue, and expenses.

        Args:
            horizon_days: Forecast horizon between 30 and 90 days.
            scenario: Optional percentage adjustments for revenue, expenses, and net cash flow.

        Returns:
            Tuple of (net_cash_flow_forecast, revenue_forecast, expenses_forecast).
        """
        horizon = self._validate_horizon(horizon_days)
        net_model, revenue_model, expense_model, _ = self._ensure_fitted()
        scenario = scenario or ForecastScenario()

        net_df = self._forecast_series(
            net_model,
            horizon,
            scenario.net_cash_flow_adjustment_pct,
            "net_cash_flow",
        )
        revenue_df = self._forecast_series(
            revenue_model,
            horizon,
            scenario.revenue_adjustment_pct,
            "revenue",
        )
        expense_df = self._forecast_series(
            expense_model,
            horizon,
            scenario.expense_adjustment_pct,
            "expenses",
        )
        return net_df, revenue_df, expense_df

    def plot_net_cash_flow_forecast(
        self,
        net_forecast: pd.DataFrame,
        historical_df: pd.DataFrame | None = None,
    ) -> go.Figure:
        """
        Build a Plotly figure for historical and forecasted net cash flow with 95% bounds.

        Returns:
            plotly.graph_objects.Figure
        """
        history = historical_df if historical_df is not None else self._historical
        if history is None:
            raise RuntimeError("Historical data unavailable. Call fit() first.")

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history["net_cash_flow"],
                mode="lines",
                name="Historical Net Cash Flow",
                line={"color": "#1f77b4", "width": 2},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=net_forecast["ds"],
                y=net_forecast["net_cash_flow_upper"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=net_forecast["ds"],
                y=net_forecast["net_cash_flow_lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.18)",
                line={"width": 0},
                name="95% Confidence Interval",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=net_forecast["ds"],
                y=net_forecast["net_cash_flow_forecast"],
                mode="lines",
                name="Forecast Net Cash Flow",
                line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
            )
        )
        figure.update_layout(
            title="Net Cash Flow Forecast (95% Confidence Interval)",
            xaxis_title="Date",
            yaxis_title="Net Cash Flow ($)",
            template="plotly_white",
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )
        return figure

    def plot_forecast_components(
        self,
        net_forecast: pd.DataFrame,
        revenue_forecast: pd.DataFrame,
        expense_forecast: pd.DataFrame,
    ) -> go.Figure:
        """
        Build a Plotly figure comparing forecasted revenue, expenses, and net cash flow.

        Returns:
            plotly.graph_objects.Figure
        """
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=revenue_forecast["ds"],
                y=revenue_forecast["revenue_forecast"],
                mode="lines",
                name="Revenue Forecast",
                line={"color": "#1f77b4", "width": 2},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=expense_forecast["ds"],
                y=expense_forecast["expenses_forecast"],
                mode="lines",
                name="Expense Forecast",
                line={"color": "#d62728", "width": 2},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=net_forecast["ds"],
                y=net_forecast["net_cash_flow_forecast"],
                mode="lines",
                name="Net Cash Flow",
                line={"color": "#2ca02c", "width": 2},
            )
        )
        figure.update_layout(
            title="Forecast Components Overview",
            xaxis_title="Date",
            yaxis_title="Amount ($)",
            template="plotly_white",
            hovermode="x unified",
        )
        return figure

    def run_forecast(
        self,
        cash_flow_df: pd.DataFrame,
        horizon_days: int = 60,
        scenario: ForecastScenario | None = None,
    ) -> ForecastResult:
        """
        Fit models, generate forecasts, and return dataframes plus Plotly figures.

        Args:
            cash_flow_df: Historical daily cash-flow data.
            horizon_days: Forecast horizon between 30 and 90 days.
            scenario: Optional scenario adjustments.

        Returns:
            ForecastResult with forecast DataFrames and Plotly figures.
        """
        self.fit(cash_flow_df)
        net_df, revenue_df, expense_df = self.forecast(horizon_days=horizon_days, scenario=scenario)
        net_figure = self.plot_net_cash_flow_forecast(net_df)
        components_figure = self.plot_forecast_components(net_df, revenue_df, expense_df)
        return ForecastResult(
            net_cash_flow=net_df,
            revenue=revenue_df,
            expenses=expense_df,
            net_cash_flow_figure=net_figure,
            components_figure=components_figure,
        )
