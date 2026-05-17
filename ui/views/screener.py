import streamlit as st
import pandas as pd
import numpy as np
import vectorbt as vbt
from core.indicators import MathEngine
from core.state_math import calculate_z_score

def render(master_matrix: pd.DataFrame, universe: dict, window_days: int = 756,
           math_engine=None, rolling_returns: pd.DataFrame = None):
    """
    Renders the Cross-Sectional Discovery Matrix (Global Screener).
    """
    st.markdown("## 🔍 Global Discovery Matrix")
    st.markdown("Top-down cross-sectional screening across segments.")

    # 1. Calculation Layer
    # Use pre-computed engine and rolling returns if provided, else compute locally
    if math_engine is None:
        math_engine = MathEngine(master_matrix)
    
    # Rolling Returns for the window
    if rolling_returns is None:
        returns_matrix = math_engine.calculate_rolling_returns(window_days)
    else:
        returns_matrix = rolling_returns
    
    # Current Metrics (Last available row)
    latest_prices = master_matrix.iloc[-1]
    latest_returns = returns_matrix.iloc[-1]
    
    # CAGR
    cagr_matrix = math_engine.calculate_cagr(returns_matrix, window_days)
    latest_cagr = cagr_matrix.iloc[-1]
    
    # Volatility (Rolling Std * sqrt(252))
    # We calculate realized volatility over the same window
    vol_matrix = returns_matrix.rolling(window=window_days).std() * np.sqrt(252)
    latest_vol = vol_matrix.iloc[-1]
    
    # Max Drawdown (manual vectorized calculation)
    window_data = master_matrix.tail(window_days)
    rolling_max = window_data.cummax()
    drawdown_matrix = (window_data - rolling_max) / rolling_max
    dd_matrix = drawdown_matrix.min()  # Worst drawdown per asset
    
    # Sharpe Ratio (Using simplified 0% Rf for now or from config if available)
    # RF is usually handled dynamically in pipeline, but here we can use a baseline
    sharpe = latest_cagr / latest_vol.replace(0, np.nan)
    
    # Z-Score (vectorized across all columns at once)
    latest_z = returns_matrix.apply(
        lambda col: calculate_z_score(col, window=window_days)
    ).iloc[-1]
    
    # 2. Build Screener Table (raw numeric values for correct sorting)
    screener_data = []
    for ticker in master_matrix.columns:
        meta = universe.get(ticker, {})
        segment = meta.get('class', 'Other').split(' - ')[0]
        
        screener_data.append({
            "Ticker": ticker,
            "Name": meta.get('name', 'N/A'),
            "Segment": segment,
            "3Y CAGR (%)": latest_cagr[ticker] * 100,
            "3Y Vol (%)": latest_vol[ticker] * 100,
            "3Y Max DD (%)": dd_matrix[ticker] * 100,
            "3Y Sharpe": sharpe[ticker],
            "Z-Score": latest_z[ticker],
            "_raw_cagr": latest_cagr[ticker],
            "_raw_z": latest_z[ticker]
        })
    
    df_screener = pd.DataFrame(screener_data)
    
    # 3. Apply Segment Ranking (numeric for sorting)
    df_screener['Segment Rank'] = df_screener.groupby('Segment')['_raw_cagr'].rank(pct=True)
    df_screener['Rank (%)'] = (df_screener['Segment Rank'] * 100).round(0)
    
    # 4. Signal Badge Logic (Simplified for Screener)
    def get_badge(row):
        if row['_raw_z'] < -1.5: return "🔴 Reversal"
        if row['_raw_z'] > 1.5: return "🟢 Momentum"
        return "🟡 Neutral"
    
    df_screener['Signal'] = df_screener.apply(get_badge, axis=1)
    
    # 5. Display Table with column formatting
    cols_to_show = ["Ticker", "Name", "Segment", "3Y CAGR (%)", "3Y Vol (%)", "3Y Max DD (%)", "3Y Sharpe", "Z-Score", "Rank (%)", "Signal"]
    
    st.markdown("### 📊 Market Opportunities")
    
    st.dataframe(
        df_screener[cols_to_show],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(
                help="NSE/BSE ticker symbol. `.NS` suffix = NSE listed."
            ),
            "Segment": st.column_config.TextColumn(
                help="Asset class grouping (Broad Market, Sectoral, Thematic, Commodities, Fixed Income). Rankings are calculated within segments."
            ),
            "3Y CAGR (%)": st.column_config.NumberColumn(
                format="%.2f%%",
                help="📈 Compound Annual Growth Rate over 3 years. Annualizes the total return. Higher = better long-term performance. >15% is strong for Indian equities."
            ),
            "3Y Vol (%)": st.column_config.NumberColumn(
                format="%.2f%%",
                help="📊 Annualized Volatility (σ) over 3 years. Measures return variability. Lower = more stable. <15% is low-risk, >30% is high-risk."
            ),
            "3Y Max DD (%)": st.column_config.NumberColumn(
                format="%.2f%%",
                help="📉 Maximum Drawdown — the worst peak-to-trough decline in 3 years. Closer to 0% = more resilient. <-30% signals significant crash risk."
            ),
            "3Y Sharpe": st.column_config.NumberColumn(
                format="%.2f",
                help="⚖️ Risk-Adjusted Return (Return ÷ Volatility). >1.0 = good, >2.0 = excellent. Compares return earned per unit of risk taken."
            ),
            "Z-Score": st.column_config.NumberColumn(
                format="%.2f",
                help="📏 Statistical deviation from historical mean. <-1.5 = historically cheap (potential value). >+1.5 = historically expensive (potential overextension). Between ±1 = normal range."
            ),
            "Rank (%)": st.column_config.NumberColumn(
                format="%.0f",
                help="🏆 Percentile rank within the asset's segment. 90 = outperforming 90% of peers. 10 = underperforming. Based on current 3Y CAGR."
            ),
            "Signal": st.column_config.TextColumn(
                help="🚦 Regime-Aware Signal:\n🟢 Momentum = top performer with positive trend.\n🟡 Neutral = within normal historical bounds.\n🔴 Reversal = deeply discounted, potential mean-reversion candidate.\nSignals are simplified here; see the Signals tab for full regime-conditioned logic."
            ),
        }
    )
    
    # Drill-down via selectbox instead of click-to-navigate
    st.markdown("---")
    st.markdown("#### 🎯 Drill-Down: Select an asset to analyze")
    drill_options = df_screener['Ticker'].tolist()
    drill_labels = [f"{row['Ticker']} — {row['Name']} ({row['Signal']})" 
                    for _, row in df_screener.iterrows()]
    
    selected_idx = st.selectbox(
        "Select asset from screener",
        range(len(drill_options)),
        format_func=lambda i: drill_labels[i],
        key="screener_drill"
    )
    
    if st.button("🔍 Drill Down to Asset Dashboard", type="primary"):
        target_asset = drill_options[selected_idx]
        st.query_params["asset"] = target_asset
        st.rerun()
