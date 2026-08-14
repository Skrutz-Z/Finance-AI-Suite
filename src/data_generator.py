"""Synthetic cash-flow and general-ledger data generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal

import numpy as np
import pandas as pd

AnomalyType = Literal["midnight_post", "high_amount", "round_number"]


@dataclass(frozen=True)
class DataGeneratorConfig:
    """Configuration for synthetic dataset generation."""

    random_seed: int = 42
    cash_flow_days: int = 365
    gl_entry_count: int = 500
    anomaly_rate: float = 0.08
    volatility: float = 0.15
    cash_flow_start_date: datetime | None = None


ACCOUNT_CODES: Final[list[str]] = [
    "1000-Cash",
    "1100-AR",
    "2000-AP",
    "4000-Revenue",
    "5000-COGS",
    "6100-Payroll",
    "6200-Rent",
    "6300-Utilities",
    "6400-Marketing",
    "6500-Travel",
    "6600-Software",
    "6700-Professional",
]

DEPARTMENTS: Final[list[str]] = [
    "Finance",
    "Operations",
    "Sales",
    "Marketing",
    "Engineering",
    "HR",
    "Legal",
    "IT",
]

VENDORS: Final[list[str]] = [
    "Acme Supplies",
    "Global Tech Partners",
    "Metro Office Co",
    "CloudServe Inc",
    "Summit Logistics",
    "Bright Media Group",
    "Northstar Consulting",
    "Prime Facilities",
    "DataFlow Systems",
    "Vertex Travel",
    "LedgerWorks",
    "BluePeak Energy",
]


def _rng(config: DataGeneratorConfig) -> np.random.Generator:
    """Return a seeded NumPy random generator."""
    return np.random.default_rng(config.random_seed)


def generate_cash_flow_data(config: DataGeneratorConfig | None = None) -> pd.DataFrame:
    """
    Generate daily cash-flow data with trend, seasonality, and noise.

    Returns a DataFrame with columns: date, revenue, expenses, net_cash_flow.
    """
    cfg = config or DataGeneratorConfig()
    generator = _rng(cfg)

    start = cfg.cash_flow_start_date or (datetime.today() - timedelta(days=cfg.cash_flow_days - 1))
    dates = pd.date_range(start=pd.Timestamp(start).normalize(), periods=cfg.cash_flow_days, freq="D")
    day_index = np.arange(cfg.cash_flow_days, dtype=float)

    # Upward trend with compounding growth.
    revenue_trend = 85000 * (1 + 0.00045 * day_index)
    expense_trend = 62000 * (1 + 0.00035 * day_index)

    # Weekly seasonality: stronger revenue mid-week, softer weekends.
    weekly_phase = 2 * np.pi * dates.dayofweek / 7
    revenue_weekly = 1.0 + 0.12 * np.sin(weekly_phase - np.pi / 2)
    expense_weekly = 1.0 + 0.06 * np.cos(weekly_phase)

    # Annual seasonality: year-end uplift and summer dip.
    annual_phase = 2 * np.pi * dates.dayofyear / 365.25
    revenue_annual = 1.0 + 0.18 * np.sin(annual_phase - np.pi / 3)
    expense_annual = 1.0 + 0.10 * np.sin(annual_phase + np.pi / 6)

    revenue_noise = generator.normal(0, cfg.volatility, cfg.cash_flow_days)
    expense_noise = generator.normal(0, cfg.volatility * 0.9, cfg.cash_flow_days)

    revenue = revenue_trend * revenue_weekly * revenue_annual * (1 + revenue_noise)
    expenses = expense_trend * expense_weekly * expense_annual * (1 + expense_noise)

    revenue = np.clip(revenue, 35000, None)
    expenses = np.clip(expenses, 25000, None)
    net_cash_flow = revenue - expenses

    return pd.DataFrame(
        {
            "date": dates,
            "revenue": np.round(revenue, 2),
            "expenses": np.round(expenses, 2),
            "net_cash_flow": np.round(net_cash_flow, 2),
        }
    )


def _sample_business_hour(generator: np.random.Generator) -> int:
    """Sample a posting hour weighted toward standard business hours."""
    hours = np.arange(24)
    weights = np.array(
        [
            0.2,
            0.2,
            0.2,
            0.3,
            0.5,
            0.8,
            1.2,
            2.5,
            4.0,
            5.5,
            6.0,
            6.5,
            6.0,
            5.5,
            5.0,
            4.5,
            4.0,
            3.0,
            1.5,
            0.8,
            0.5,
            0.4,
            0.3,
            0.2,
        ],
        dtype=float,
    )
    return int(generator.choice(hours, p=weights / weights.sum()))


def _inject_gl_anomalies(
    df: pd.DataFrame,
    generator: np.random.Generator,
    anomaly_rate: float,
) -> pd.DataFrame:
    """Inject midnight posts, high amounts, and round-number anomalies."""
    result = df.copy()
    anomaly_count = max(1, int(len(result) * anomaly_rate))
    anomaly_indices = generator.choice(len(result), size=anomaly_count, replace=False)
    anomaly_types = generator.choice(
        ["midnight_post", "high_amount", "round_number"],
        size=anomaly_count,
        p=[0.35, 0.35, 0.30],
    )

    for idx, anomaly_type in zip(anomaly_indices, anomaly_types):
        if anomaly_type == "midnight_post":
            result.at[idx, "Posting_Hour"] = int(generator.integers(0, 3))
        elif anomaly_type == "high_amount":
            multiplier = float(generator.uniform(8.0, 18.0))
            result.at[idx, "Amount"] = round(float(result.at[idx, "Amount"]) * multiplier, 2)
        else:
            base = float(generator.choice([5000, 10000, 25000, 50000, 100000]))
            sign = -1.0 if float(result.at[idx, "Amount"]) < 0 else 1.0
            result.at[idx, "Amount"] = round(sign * base, 2)

    return result


def generate_gl_journal_entries(config: DataGeneratorConfig | None = None) -> pd.DataFrame:
    """
    Generate synthetic general-ledger journal entries with injected anomalies.

    Returns a DataFrame with columns:
    Transaction_ID, Amount, Posting_Hour, Account_Code, Department, Vendor.
    """
    cfg = config or DataGeneratorConfig()
    generator = _rng(cfg)

    transaction_ids = [f"JE-{100000 + i:06d}" for i in range(cfg.gl_entry_count)]
    posting_hours = [_sample_business_hour(generator) for _ in range(cfg.gl_entry_count)]
    account_codes = generator.choice(ACCOUNT_CODES, size=cfg.gl_entry_count)
    departments = generator.choice(DEPARTMENTS, size=cfg.gl_entry_count)
    vendors = generator.choice(VENDORS, size=cfg.gl_entry_count)

    # Log-normal amounts centered around typical AP/operating spend.
    base_amounts = generator.lognormal(mean=7.2, sigma=0.55, size=cfg.gl_entry_count)
    signs = generator.choice([-1, 1], size=cfg.gl_entry_count, p=[0.42, 0.58])
    amounts = np.round(base_amounts * signs, 2)

    df = pd.DataFrame(
        {
            "Transaction_ID": transaction_ids,
            "Amount": amounts,
            "Posting_Hour": posting_hours,
            "Account_Code": account_codes,
            "Department": departments,
            "Vendor": vendors,
        }
    )

    return _inject_gl_anomalies(df, generator, cfg.anomaly_rate)


def generate_all_datasets(config: DataGeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate both cash-flow and general-ledger datasets.

    Returns:
        Tuple of (cash_flow_df, gl_journal_df).
    """
    cfg = config or DataGeneratorConfig()
    cash_flow = generate_cash_flow_data(cfg)
    gl_entries = generate_gl_journal_entries(cfg)
    return cash_flow, gl_entries


if __name__ == "__main__":
    cash_flow_df, gl_df = generate_all_datasets()
    print("Cash Flow Preview:")
    print(cash_flow_df.head())
    print("\nGeneral Ledger Preview:")
    print(gl_df.head())
    print(f"\nCash flow rows: {len(cash_flow_df)}")
    print(f"GL entries: {len(gl_df)}")
