"""Enterprise Finance AI Suite — Streamlit dashboard."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import streamlit as st

from src.anomaly_detector import AuditAnomalyDetector
from src.data_generator import DataGeneratorConfig, generate_all_datasets
from src.financial_rag import SAMPLE_10K_NOTES, FinancialCopilot
from src.forecaster import CashFlowForecaster, ForecastScenario

DEMO_WATERMARK = "This is not the final product, just a demo made by Shubh Nahar"

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Enterprise Finance AI Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3.5rem; }
    .suite-header {
        background: linear-gradient(135deg, #0f2744 0%, #1a3a5c 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }
    .suite-header h1 { color: #ffffff !important; margin-bottom: 0.25rem; font-size: 1.85rem; }
    .suite-header p  { color: #c8d9ec !important; margin: 0; font-size: 0.95rem; }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(15, 39, 68, 0.06);
    }
    div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.8rem; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f2744 !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.25rem;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    .doc-viewer {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #111827;
        max-height: 320px;
        overflow-y: auto;
    }
    .chat-source {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    .demo-watermark {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999;
        text-align: center;
        padding: 0.55rem 1rem;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: #64748b;
        background: rgba(248, 250, 252, 0.92);
        border-top: 1px solid #e2e8f0;
        backdrop-filter: blur(4px);
    }
    .demo-watermark-sidebar {
        margin-top: 2rem;
        padding: 0.75rem;
        text-align: center;
        font-size: 0.72rem;
        line-height: 1.4;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_currency(value: float) -> str:
    """Format a numeric value as USD currency."""
    return f"${value:,.0f}"


def render_demo_watermark() -> None:
    """Render a persistent demo disclaimer watermark at the bottom of the page."""
    st.markdown(
        f"<div class='demo-watermark'>{DEMO_WATERMARK}</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_datasets(volatility: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and cache synthetic cash-flow and GL datasets."""
    config = DataGeneratorConfig(volatility=volatility, random_seed=seed)
    return generate_all_datasets(config)


def init_session_state() -> None:
    """Initialize Streamlit session state defaults."""
    defaults: dict[str, Any] = {
        "chat_history": [],
        "anomaly_count": 0,
        "copilot_provider": "local",
        "detection_result": None,
        "forecast_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_kpi_cards(total_inflow: float, anomalies: int, copilot_status: str) -> None:
    """Render header KPI metric cards."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Inflow (LTM Revenue)", format_currency(total_inflow), delta="365-day synthetic")
    with col2:
        st.metric("Anomalies Detected", str(anomalies), delta="GL audit scan", delta_color="inverse")
    with col3:
        status_delta = "Ready" if copilot_status != "local" else "Local mode"
        st.metric("Copilot Status", copilot_status.upper(), delta=status_delta)


def build_variance_summary(copilot: FinancialCopilot) -> pd.DataFrame:
    """Build a variance breakdown table from sample copilot questions."""
    rows: list[dict[str, str]] = []
    for question in copilot.get_sample_questions():
        try:
            response = copilot.ask(question, force_local=True)
            top_source = response.sources[0].chunk.title if response.sources else "N/A"
            preview = response.answer.split("\n")[2] if len(response.answer.split("\n")) > 2 else response.answer[:160]
            rows.append(
                {
                    "Variance Topic": question,
                    "Primary Disclosure": top_source,
                    "Key Driver Summary": preview[:180],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Variance Topic": question,
                    "Primary Disclosure": "Error",
                    "Key Driver Summary": str(exc),
                }
            )
    return pd.DataFrame(rows)


def get_copilot() -> FinancialCopilot:
    """Return an indexed FinancialCopilot configured from sidebar settings."""
    openai_key = st.session_state.get("openai_api_key") or None
    gemini_key = st.session_state.get("gemini_api_key") or None
    ollama_model = st.session_state.get("ollama_model") or None
    if openai_key:
        st.session_state.copilot_provider = "openai"
    elif gemini_key:
        st.session_state.copilot_provider = "gemini"
    elif ollama_model:
        st.session_state.copilot_provider = "ollama"
    else:
        st.session_state.copilot_provider = "local"

    copilot = FinancialCopilot(
        openai_api_key=openai_key if openai_key else None,
        gemini_api_key=gemini_key if gemini_key else None,
        ollama_model=ollama_model if ollama_model else None,
    )
    copilot.index_10k_notes()
    return copilot


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[float, int]:
    """Render global sidebar controls and return volatility plus seed."""
    st.sidebar.title("Control Center")
    st.sidebar.markdown("Configure data, models, and optional LLM providers.")

    volatility = st.sidebar.slider(
        "Data Volatility",
        min_value=0.05,
        max_value=0.35,
        value=0.15,
        step=0.01,
        help="Controls noise in synthetic cash-flow generation.",
    )
    seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)

    st.sidebar.divider()
    st.sidebar.subheader("LLM Providers (Optional)")
    st.session_state.openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )
    st.session_state.gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
    )
    st.session_state.ollama_model = st.sidebar.text_input(
        "Ollama Model",
        placeholder="llama3.2 (local)",
    )

    if st.sidebar.button("Refresh Datasets", use_container_width=True):
        load_datasets.clear()
        st.session_state.detection_result = None
        st.session_state.forecast_result = None
        st.rerun()

    st.sidebar.markdown(
        f"<div class='demo-watermark-sidebar'>{DEMO_WATERMARK}</div>",
        unsafe_allow_html=True,
    )

    return float(volatility), int(seed)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def render_forecaster_tab(cash_flow_df: pd.DataFrame) -> None:
    """Render the Cash Flow Forecaster tab."""
    st.subheader("Cash Flow Forecaster")
    st.caption("Prophet-based forecasting with scenario adjustments and 95% confidence intervals.")

    control_col, metric_col = st.columns([1.2, 2])

    with control_col:
        st.markdown("##### Forecast Controls")
        horizon = st.slider("Forecast Horizon (days)", min_value=30, max_value=90, value=60, step=5)
        high_volatility = st.toggle(
            "High-Volatility Scenario",
            value=False,
            help="Applies pessimistic net-cash-flow and expense stress adjustments.",
        )
        revenue_adj = st.slider("Revenue Adjustment (%)", -20, 20, 0, 1)
        expense_adj = st.slider("Expense Adjustment (%)", -20, 20, 0, 1)
        net_adj = st.slider("Net Cash Flow Adjustment (%)", -20, 20, 0, 1)
        run_forecast = st.button("Run Forecast", type="primary", use_container_width=True)

    scenario = ForecastScenario(
        revenue_adjustment_pct=float(revenue_adj),
        expense_adjustment_pct=float(expense_adj + (-5 if high_volatility else 0)),
        net_cash_flow_adjustment_pct=float(net_adj + (-8 if high_volatility else 0)),
    )

    if run_forecast or st.session_state.forecast_result is None:
        with st.spinner("Training Prophet models and generating forecast..."):
            try:
                forecaster = CashFlowForecaster()
                st.session_state.forecast_result = forecaster.run_forecast(
                    cash_flow_df,
                    horizon_days=horizon,
                    scenario=scenario,
                )
            except Exception as exc:
                st.error(f"Forecast failed: {exc}")
                return

    result = st.session_state.forecast_result
    if result is None:
        st.info("Click **Run Forecast** to generate predictions.")
        return

    net_df = result.net_cash_flow
    avg_forecast = net_df["net_cash_flow_forecast"].mean()
    lower_band = net_df["net_cash_flow_lower"].mean()
    upper_band = net_df["net_cash_flow_upper"].mean()

    with metric_col:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Forecast NCF", format_currency(avg_forecast))
        m2.metric("Lower CI (95%)", format_currency(lower_band))
        m3.metric("Upper CI (95%)", format_currency(upper_band))
        m4.metric("Horizon", f"{len(net_df)} days")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(result.net_cash_flow_figure, use_container_width=True)
    with chart_col2:
        st.plotly_chart(result.components_figure, use_container_width=True)

    with st.expander("Forecast Data Preview", expanded=False):
        preview = net_df.copy()
        preview["ds"] = preview["ds"].dt.strftime("%Y-%m-%d")
        st.dataframe(preview, use_container_width=True, hide_index=True)


def render_anomaly_tab(gl_df: pd.DataFrame) -> None:
    """Render the GL Anomaly Detector tab."""
    st.subheader("GL Anomaly Detector")
    st.caption("Isolation Forest audit scan with configurable sensitivity and export.")

    ctrl_col, _ = st.columns([1, 2.5])
    with ctrl_col:
        sensitivity = st.slider(
            "Detection Sensitivity (contamination)",
            min_value=0.02,
            max_value=0.20,
            value=0.08,
            step=0.01,
            help="Higher values flag more transactions as anomalous.",
        )
        top_n = st.slider("Summary Table Size", min_value=5, max_value=50, value=15, step=5)
        run_detection = st.button("Run Audit Scan", type="primary", use_container_width=True)

    if run_detection or st.session_state.detection_result is None:
        with st.spinner("Scoring journal entries for anomalies..."):
            try:
                detector = AuditAnomalyDetector(contamination=float(sensitivity))
                st.session_state.detection_result = detector.run_detection(gl_df, top_n=int(top_n))
                st.session_state.anomaly_count = int(
                    st.session_state.detection_result.scored_entries["anomaly_flag"].sum()
                )
            except Exception as exc:
                st.error(f"Anomaly detection failed: {exc}")
                return

    result = st.session_state.detection_result
    if result is None:
        st.info("Click **Run Audit Scan** to analyze journal entries.")
        return

    scored = result.scored_entries
    flagged = scored[scored["anomaly_flag"]].copy()
    flagged_count = len(flagged)
    medium_count = int((scored["risk_level"] == "Medium").sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Flagged Transactions", flagged_count)
    m2.metric("Medium Risk", medium_count)
    m3.metric("Total Entries Scanned", len(scored))

    st.plotly_chart(result.scatter_figure, use_container_width=True)

    st.markdown("##### Flagged Transactions — Audit Table")
    display_cols = [
        "Transaction_ID",
        "Amount",
        "Posting_Hour",
        "Account_Code",
        "Department",
        "Vendor",
        "anomaly_score",
        "risk_level",
    ]
    audit_table = flagged[display_cols].sort_values("anomaly_score").reset_index(drop=True)
    st.dataframe(audit_table, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    flagged.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Flagged Transactions (CSV)",
        data=csv_buffer.getvalue(),
        file_name="flagged_gl_transactions.csv",
        mime="text/csv",
        use_container_width=False,
    )


def render_copilot_tab() -> None:
    """Render the Financial Variance Copilot tab."""
    st.subheader("Financial Variance Copilot")
    st.caption("10-K note retrieval, Q&A chat, and variance breakdown summaries.")

    try:
        copilot = get_copilot()
    except Exception as exc:
        st.error(f"Copilot initialization failed: {exc}")
        return

    doc_col, chat_col = st.columns([1, 1.4])

    with doc_col:
        st.markdown("##### 10-K Document Viewer")
        note_titles = [f"{note['section']} — {note['title']}" for note in SAMPLE_10K_NOTES]
        selected = st.selectbox("Select disclosure", note_titles, label_visibility="collapsed")
        selected_index = note_titles.index(selected)
        note = SAMPLE_10K_NOTES[selected_index]
        st.markdown(
            f"<div class='doc-viewer'><strong>{note['section']} | {note['title']}</strong><br><br>"
            f"{note['content']}</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Variance Breakdown Summary", expanded=True):
            with st.spinner("Summarizing variance drivers..."):
                summary_df = build_variance_summary(copilot)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with chat_col:
        st.markdown("##### Copilot Q&A")
        st.caption("Sample prompts or ask your own variance question.")

        sample_cols = st.columns(2)
        for index, question in enumerate(copilot.get_sample_questions()):
            col = sample_cols[index % 2]
            if col.button(question, key=f"sample_q_{index}", use_container_width=True):
                st.session_state.pending_question = question

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    st.markdown(
                        f"<div class='chat-source'>Sources: {', '.join(message['sources'])}</div>",
                        unsafe_allow_html=True,
                    )

        user_input = st.chat_input("Ask about SG&A, gross margin, FX, or other variances...")
        question = st.session_state.pop("pending_question", None) or user_input

        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving 10-K context..."):
                    try:
                        response = copilot.ask(question)
                        st.markdown(response.answer)
                        source_labels = [item.chunk.source_label for item in response.sources]
                        if source_labels:
                            st.markdown(
                                f"<div class='chat-source'>Sources: {', '.join(source_labels)} "
                                f"| Provider: {response.provider}</div>",
                                unsafe_allow_html=True,
                            )
                        st.session_state.chat_history.append(
                            {
                                "role": "user",
                                "content": question,
                            }
                        )
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": response.answer,
                                "sources": source_labels,
                            }
                        )
                    except Exception as exc:
                        st.error(f"Copilot query failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Enterprise Finance AI Suite dashboard."""
    init_session_state()
    volatility, seed = render_sidebar()

    try:
        cash_flow_df, gl_df = load_datasets(volatility, seed)
    except Exception as exc:
        st.error(f"Failed to load datasets: {exc}")
        return

    total_inflow = float(cash_flow_df["revenue"].sum())

    if st.session_state.detection_result is None:
        try:
            detector = AuditAnomalyDetector()
            st.session_state.detection_result = detector.run_detection(gl_df)
            st.session_state.anomaly_count = int(
                st.session_state.detection_result.scored_entries["anomaly_flag"].sum()
            )
        except Exception:
            st.session_state.anomaly_count = 0

    st.markdown(
        """
        <div class="suite-header">
            <h1>Enterprise Finance AI Suite</h1>
            <p>Cash-flow forecasting, GL anomaly detection, and 10-K variance copilot</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_cards(
        total_inflow=total_inflow,
        anomalies=st.session_state.anomaly_count,
        copilot_status=st.session_state.copilot_provider,
    )

    tab_forecast, tab_anomaly, tab_copilot = st.tabs(
        ["Cash Flow Forecaster", "GL Anomaly Detector", "Financial Variance Copilot"]
    )

    with tab_forecast:
        render_forecaster_tab(cash_flow_df)

    with tab_anomaly:
        render_anomaly_tab(gl_df)

    with tab_copilot:
        render_copilot_tab()

    render_demo_watermark()


if __name__ == "__main__":
    main()
