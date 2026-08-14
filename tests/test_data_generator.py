"""Unit tests for synthetic data generation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_generator import (
    DataGeneratorConfig,
    generate_all_datasets,
    generate_cash_flow_data,
    generate_gl_journal_entries,
)


class TestCashFlowGeneration:
    """Tests for daily cash-flow synthetic data."""

    def test_output_shape_and_columns(self, default_config: DataGeneratorConfig) -> None:
        df = generate_cash_flow_data(default_config)
        assert len(df) == default_config.cash_flow_days
        assert list(df.columns) == ["date", "revenue", "expenses", "net_cash_flow"]

    def test_net_cash_flow_identity(self, cash_flow_df: pd.DataFrame) -> None:
        expected = cash_flow_df["revenue"] - cash_flow_df["expenses"]
        pd.testing.assert_series_equal(
            cash_flow_df["net_cash_flow"],
            expected,
            check_names=False,
        )

    def test_revenue_and_expenses_are_positive(self, cash_flow_df: pd.DataFrame) -> None:
        assert (cash_flow_df["revenue"] > 0).all()
        assert (cash_flow_df["expenses"] > 0).all()

    def test_dates_are_monotonic(self, cash_flow_df: pd.DataFrame) -> None:
        dates = pd.to_datetime(cash_flow_df["date"])
        assert dates.is_monotonic_increasing

    def test_reproducibility_with_seed(self) -> None:
        config = DataGeneratorConfig(random_seed=7, cash_flow_days=30)
        first = generate_cash_flow_data(config)
        second = generate_cash_flow_data(config)
        pd.testing.assert_frame_equal(first, second)


class TestGLJournalGeneration:
    """Tests for general-ledger journal entry synthesis."""

    def test_output_shape_and_columns(self, default_config: DataGeneratorConfig) -> None:
        df = generate_gl_journal_entries(default_config)
        assert len(df) == default_config.gl_entry_count
        assert set(df.columns) == {
            "Transaction_ID",
            "Amount",
            "Posting_Hour",
            "Account_Code",
            "Department",
            "Vendor",
        }

    def test_transaction_ids_are_unique(self, gl_df: pd.DataFrame) -> None:
        assert gl_df["Transaction_ID"].is_unique

    def test_posting_hours_in_valid_range(self, gl_df: pd.DataFrame) -> None:
        assert gl_df["Posting_Hour"].between(0, 23).all()

    def test_injected_anomaly_signals_exist(self, gl_df: pd.DataFrame) -> None:
        midnight_posts = (gl_df["Posting_Hour"] <= 2).sum()
        round_amounts = (gl_df["Amount"].abs() % 1000 == 0).sum()
        assert midnight_posts >= 1
        assert round_amounts >= 1

    def test_reproducibility_with_seed(self) -> None:
        config = DataGeneratorConfig(random_seed=99, gl_entry_count=50)
        first = generate_gl_journal_entries(config)
        second = generate_gl_journal_entries(config)
        pd.testing.assert_frame_equal(first, second)


class TestGenerateAllDatasets:
    """Tests for combined dataset generation."""

    def test_returns_two_dataframes(self, datasets: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        cash_flow, gl_entries = datasets
        assert isinstance(cash_flow, pd.DataFrame)
        assert isinstance(gl_entries, pd.DataFrame)
        assert not cash_flow.empty
        assert not gl_entries.empty

    def test_volatility_affects_output(self) -> None:
        low_vol = generate_cash_flow_data(DataGeneratorConfig(random_seed=1, volatility=0.05, cash_flow_days=60))
        high_vol = generate_cash_flow_data(DataGeneratorConfig(random_seed=1, volatility=0.30, cash_flow_days=60))
        low_std = low_vol["net_cash_flow"].std()
        high_std = high_vol["net_cash_flow"].std()
        assert high_std > low_std
