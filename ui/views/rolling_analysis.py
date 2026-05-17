import streamlit as st
from data.fetcher import fetch_delta, load_master_data
from core.universe import get_universe_tickers

def render_rolling_analysis_view(universe_name, strategy_name, date_range):
    st.header(f"🔭 Market Screener: {strategy_name}")
    st.caption(f"Analyzing {universe_name} universe from {date_range[0]} to {date_range[1]}")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🚀 Run Full Scan"):
            tickers = get_universe_tickers(universe_name)
            with st.spinner(f"Syncing data for {len(tickers)} assets..."):
                fetch_delta(tickers)
            st.success("Data synced! (Placeholder for actual backtest execution)")
            
    with col2:
        st.info("The screen summary will appear here after execution.")
