import streamlit as st
import pandas as pd
import numpy as np
import json
import logging
import sys
from pathlib import Path
import plotly.io as pio
import vectorbt as vbt

# Fix relative imports when executing from root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import CONFIG_DIR, STREAMLIT_TITLE, STREAMLIT_LAYOUT, TRADING_DAYS_PER_YEAR
from data.pipeline import DataPipeline
from core.indicators import MathEngine
from core.validators import validate_matrix_shape, validate_inception_alignment
from ui.components import (
    render_kpi_dashboard, 
    render_timeseries_chart, 
    render_distribution_matrix,
    render_methodology_drilldown
)
from ui.help_guide import render_help_guide, render_sidebar_help

st.set_page_config(page_title=STREAMLIT_TITLE, layout=STREAMLIT_LAYOUT, page_icon="🛡️")

@st.cache_data
def load_universe():
    with open(CONFIG_DIR / "universe.json", 'r') as f:
        return json.load(f)

@st.cache_data
def load_and_validate_matrix_v2():
    pipeline = DataPipeline()
    matrix = pipeline.build_primary_matrix()
    uni = load_universe()
    validate_matrix_shape(matrix, uni)
    validate_inception_alignment(matrix, uni)
    return matrix

@st.cache_data
def compute_rolling_returns(_matrix, window_days: int):
    """Cached rolling returns computation — computed ONCE per window change."""
    engine = MathEngine(_matrix)
    return engine.calculate_rolling_returns(window_days)

def main():
    st.title("🛡️ " + STREAMLIT_TITLE)

    universe = load_universe()

    try:
        master_matrix = load_and_validate_matrix_v2()
    except Exception as e:
        st.error(f"Failed to initialize Matrix Engine: {e}")
        st.info("Have you fetched the initial data using data_fetcher.py?")
        return

    math_engine = MathEngine(master_matrix)

    # ── Sidebar ─────────────────────────────────────────────────
    st.sidebar.header("Control Panel")

    # --- 1. Asset Class Filter ---
    all_classes = set()
    for meta in universe.values():
        broad_class = meta.get('class', 'Other').split(' - ')[0]
        all_classes.add(broad_class)
    
    selected_class = st.sidebar.selectbox("Filter Asset Class", ["All"] + sorted(list(all_classes)))

    # Sort and filter universe by class and inception date
    filtered_tickers = []
    for t in universe.keys():
        broad_class = universe[t].get('class', 'Other').split(' - ')[0]
        if selected_class == "All" or broad_class == selected_class:
            filtered_tickers.append(t)

    sorted_tickers = sorted(
        filtered_tickers, 
        key=lambda x: pd.to_datetime(universe[x].get('inception', '2099-01-01'))
    )

    # Check for URL override
    default_ticker_idx = 0
    query_asset = st.query_params.get("asset")
    if query_asset in sorted_tickers:
        default_ticker_idx = sorted_tickers.index(query_asset)

    primary_ticker = st.sidebar.selectbox(
        "Primary Asset Selector",
        options=sorted_tickers,
        index=default_ticker_idx,
        format_func=lambda x: f"{x} - {universe[x]['name']}"
    )

    benchmarks = st.sidebar.multiselect(
        "Benchmark Overlays",
        options=[t for t in universe.keys() if t != primary_ticker],
        format_func=lambda x: universe[x]['name']
    )

    # --- 2. Historical Lookback Slider ---
    min_date = master_matrix.index.min().to_pydatetime()
    max_date = master_matrix.index.max().to_pydatetime()
    lookback_range = st.sidebar.slider(
        "Historical Lookback",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MMM YYYY"
    )

    # Predefined rolling windows
    window_map = {
        "1M (21d)": 21,
        "3M (63d)": 63,
        "6M (126d)": 126,
        "1Y (252d)": 252,
        "3Y (756d)": 756,
        "5Y (1260d)": 1260
    }
    window_label = st.sidebar.radio("Rolling Window", list(window_map.keys()), index=3)
    window_days = window_map[window_label]

    # Sidebar quick reference
    render_sidebar_help()

    # ── Main Tabs ─────────────────────────────────────────────────
    tab_screen, tab_analysis, tab_exec, tab_help = st.tabs(["🔍 Global Screener", "📊 Asset Dashboard", "🎯 Signals", "📖 Help & Guide"])

    # Handle URL Parameters for deep linking
    query_asset = st.query_params.get("asset")
    if query_asset and query_asset in sorted_tickers:
        # We don't force it here because the selectbox has its own state,
        # but we can use it to suggest the user is in drill-down mode.
        pass

    # Calculate rolling returns matrix ONCE for all tabs (cached)
    rolling_returns = compute_rolling_returns(master_matrix, window_days)

    # ── TAB 0: GLOBAL SCREENER ────────────────────────────────────
    with tab_screen:
        from ui.views import screener
        screener.render(
            master_matrix=master_matrix,
            universe=universe,
            window_days=window_days,
            math_engine=math_engine,
            rolling_returns=rolling_returns
        )

    # ── TAB 1: ASSET DASHBOARD ────────────────────────────────────
    with tab_analysis:
        from ui.views import asset_dashboard
        asset_dashboard.render(
            master_matrix=master_matrix,
            universe=universe,
            primary_ticker=primary_ticker,
            benchmarks=benchmarks,
            math_engine=math_engine,
            window_days=window_days,
            window_label=window_label,
            lookback_range=lookback_range,
            rolling_returns=rolling_returns
        )

    # ── TAB 2: SIGNALS (EXECUTION LAYER) ────────────────────────────
    with tab_exec:
        from ui.views import signals
        signals.render(
            master_matrix=master_matrix,
            primary_ticker=primary_ticker,
            rolling_returns_matrix=rolling_returns,
            universe=universe
        )

    # ── TAB 3: HELP GUIDE ─────────────────────────────────────────
    with tab_help:
        render_help_guide()

if __name__ == "__main__":
    main()
