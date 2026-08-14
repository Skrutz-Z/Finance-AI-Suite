"""Shared pytest fixtures for the Enterprise Finance AI Suite."""

from __future__ import annotations

import pytest

from src.data_generator import DataGeneratorConfig, generate_all_datasets, generate_cash_flow_data, generate_gl_journal_entries


@pytest.fixture
def default_config() -> DataGeneratorConfig:
    """Return a deterministic generator configuration for tests."""
    return DataGeneratorConfig(
        random_seed=42,
        cash_flow_days=120,
        gl_entry_count=200,
        anomaly_rate=0.10,
        volatility=0.12,
    )


@pytest.fixture
def cash_flow_df(default_config: DataGeneratorConfig):
    """Return a synthetic cash-flow DataFrame."""
    return generate_cash_flow_data(default_config)


@pytest.fixture
def gl_df(default_config: DataGeneratorConfig):
    """Return synthetic general-ledger journal entries."""
    return generate_gl_journal_entries(default_config)


@pytest.fixture
def datasets(default_config: DataGeneratorConfig):
    """Return both synthetic datasets as a tuple."""
    return generate_all_datasets(default_config)
