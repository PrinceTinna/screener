import streamlit as st
import pandas as pd
import numpy as np
import vectorbt as vbt
from core.indicators import MathEngine
from core.state_math import calculate_z_score, calculate_bubble_z_score, classify_bubble_status
from config.settings import RISK_FREE_RATE

def render(master_matrix: pd.DataFrame, universe: dict, window_days: int = 756,
           math_engine=None, rolling_returns: pd.DataFrame = None,
           pe_matrix: pd.DataFrame = None, eps_matrix: pd.DataFrame = None):
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
    
    # Volatility (Annualized from DAILY returns, not from rolling N-day returns)
    daily_returns = master_matrix.pct_change()
    tail_daily = daily_returns.tail(window_days)
    latest_vol = tail_daily.std() * np.sqrt(252)
    latest_vol[tail_daily.count() < window_days // 2] = np.nan
    
    # Max Drawdown (manual vectorized calculation)
    window_data = master_matrix.tail(window_days)
    rolling_max = window_data.cummax()
    drawdown_matrix = (window_data - rolling_max) / rolling_max
    dd_matrix = drawdown_matrix.min()  # Worst drawdown per asset
    
    # Sharpe Ratio (using RISK_FREE_RATE from settings — India 10Y G-Sec proxy)
    sharpe = (latest_cagr - RISK_FREE_RATE) / latest_vol.replace(0, np.nan)
    
    # Z-Score — use calculate_z_score (same function as Signals tab) for consistency
    # Compute per-column on the rolling returns series, take the latest value
    latest_z = pd.Series(np.nan, index=master_matrix.columns)
    for col in returns_matrix.columns:
        col_series = returns_matrix[col].dropna()
        if len(col_series) >= 252:
            z_series = calculate_z_score(col_series, window=252)
            if len(z_series.dropna()) > 0:
                latest_z[col] = z_series.iloc[-1]
    
    # 2. Build Screener Table (raw numeric values for correct sorting)
    screener_data = []
    for ticker in master_matrix.columns:
        meta = universe.get(ticker, {})
        segment = meta.get('class', 'Other').split(' - ')[0]
        
        # Calculate Bubble Z-Score using lifetime history of the asset (with Check B: 1000 days history guard)
        price_series = master_matrix[ticker].dropna()
        if len(price_series) < 1000:
            bubble_class = {
                "label": "🟡 Insufficient History"
            }
        else:
            bubble_z_series = calculate_bubble_z_score(price_series)
            latest_bubble_z = bubble_z_series.iloc[-1] if not bubble_z_series.empty else np.nan
            bubble_class = classify_bubble_status(latest_bubble_z)
            
        # Check D: Cross-Sectional Synchronization Guard
        from config.settings import CACHE_DIR
        raw_path = CACHE_DIR / f"{ticker}_raw.parquet"
        is_stale = False
        if raw_path.exists():
            try:
                last_raw_date = pd.read_parquet(raw_path, columns=[]).index[-1]
                last_raw_date = pd.to_datetime(last_raw_date)
                target_date = master_matrix.index[-1]
                # Exclude if it's lagging by more than 3 business days (~5 calendar days)
                if (target_date - last_raw_date).days > 5:
                    is_stale = True
            except Exception:
                is_stale = True
        
        # If the ticker is stale or has less than 252 days of history, exclude its return from ranking
        raw_cagr = latest_cagr[ticker]
        if is_stale or len(price_series) < 252:
            raw_cagr = np.nan
            
        # Get PE and EPS from matrices
        latest_pe = np.nan
        eps_growth = np.nan
        if pe_matrix is not None and ticker in pe_matrix.columns:
            latest_pe = pe_matrix[ticker].iloc[-1]
            
        if eps_matrix is not None and ticker in eps_matrix.columns:
            ticker_eps = eps_matrix[ticker].dropna()
            if len(ticker_eps) >= 252:
                current_eps = ticker_eps.iloc[-1]
                prev_eps = ticker_eps.iloc[-252]
                if prev_eps > 0:
                    eps_growth = ((current_eps / prev_eps) - 1.0) * 100
        
        screener_data.append({
            "Ticker": ticker,
            "Name": meta.get('name', 'N/A'),
            "Segment": segment,
            "3Y CAGR (%)": latest_cagr[ticker] * 100,
            "3Y Vol (%)": latest_vol[ticker] * 100,
            "3Y Max DD (%)": dd_matrix[ticker] * 100,
            "3Y Sharpe": sharpe[ticker],
            "Z-Score": latest_z[ticker],
            "Bubble Risk": bubble_class["label"],
            "P/E Ratio (Est.)": latest_pe,
            "YoY EPS Growth (Est. %)": eps_growth,
            "_raw_cagr": raw_cagr,
            "_raw_z": latest_z[ticker]
        })
    
    df_screener = pd.DataFrame(screener_data)
    
    # 3. Apply Segment Ranking (numeric for sorting)
    df_screener['Segment Rank'] = df_screener.groupby('Segment')['_raw_cagr'].rank(pct=True)
    df_screener['Rank (%)'] = (df_screener['Segment Rank'] * 100).round(0)
    
    # 4. Signal Badge Logic (Simplified for Screener)
    def get_badge(row):
        ticker = row['Ticker']
        price_series = master_matrix[ticker].dropna()
        if len(price_series) < 252:
            return "🟡 Insufficient History"
            
        # Check C: Fat-Tail Outlier CAGR Cap
        cagr = row['_raw_cagr']
        if pd.isna(cagr) or cagr > 1.50:
            if not pd.isna(cagr) and cagr > 1.50:
                price_slice = price_series.tail(252)
                missing_pct = price_slice.isna().sum() / 252
                if missing_pct > 0.05:
                    return "🔴 High Uncertainty"
                    
        if row['_raw_z'] < -1.5: return "🔴 Reversal"
        if row['_raw_z'] > 1.5: return "🟢 Momentum"
        return "🟡 Neutral"
    
    df_screener['Signal'] = df_screener.apply(get_badge, axis=1)
    
    # 5. Display Table with column formatting
    cols_to_show = ["Ticker", "Name", "Segment", "3Y CAGR (%)", "3Y Vol (%)", "3Y Max DD (%)", "3Y Sharpe", "Z-Score", "Bubble Risk", "P/E Ratio (Est.)", "YoY EPS Growth (Est. %)", "Rank (%)", "Signal"]
    
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
                help=f"⚖️ Risk-Adjusted Return ((CAGR - Rf) ÷ Volatility). Rf = {RISK_FREE_RATE*100:.0f}% (India 10Y G-Sec). >1.0 = good, >2.0 = excellent."
            ),
            "Z-Score": st.column_config.NumberColumn(
                format="%.2f",
                help="📏 Statistical deviation from historical mean. <-1.5 = historically cheap (potential value). >+1.5 = historically expensive (potential overextension). Between ±1 = normal range."
            ),
            "Rank (%)": st.column_config.NumberColumn(
                format="%.0f",
                help="🏆 Percentile rank within the asset's segment. 90 = outperforming 90% of peers. 10 = underperforming. Based on current 3Y CAGR."
            ),
            "Bubble Risk": st.column_config.TextColumn(
                help="🫧 Bubble Risk level (lifetime 3Y detrended price distance Z-score):\n🟢 Normal = Z < 1.5\n🟡 Extended = 1.5 <= Z < 2.0\n🟠 2-Sigma Bubble = 2.0 <= Z < 3.0 (Outlier)\n🔴 3-Sigma Superbubble = Z >= 3.0 (Extreme)"
            ),
            "P/E Ratio (Est.)": st.column_config.NumberColumn(
                format="%.2f",
                help="🏷️ **Model-Estimated** Price-to-Earnings Ratio (not from live filings). Derived from historical PE benchmarks and price interpolation. High = growth/expensive. Low = value/cheap. Commodities/Cash are NaN."
            ),
            "YoY EPS Growth (Est. %)": st.column_config.NumberColumn(
                format="%.2f%%",
                help="📈 **Model-Estimated** Year-over-Year earnings-per-share growth. Derived from synthetic fundamental model, NOT from corporate filings. Treat as directional only."
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
