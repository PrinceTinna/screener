import streamlit as st
import pandas as pd
import numpy as np

from ui.components import (
    render_kpi_dashboard, 
    render_timeseries_chart, 
    render_distribution_matrix,
    render_methodology_drilldown
)
from config.settings import TRADING_DAYS_PER_YEAR
from core.state_math import classify_regime_2d, calculate_bubble_z_score, classify_bubble_status, classify_bubble_series
from core.validator import VAL_CROSS_CHECK_MAP

@st.cache_data
def _cached_vbt_stats(_daily_returns):
    """Cache VBT stats per ticker — independent of rolling window selection."""
    return _daily_returns.vbt.returns.stats(settings=dict(freq='D'))

def render(master_matrix: pd.DataFrame, universe: dict, primary_ticker: str, benchmarks: list, 
           math_engine, window_days: int, window_label: str, lookback_range: tuple,
           rolling_returns: pd.DataFrame = None, pe_matrix: pd.DataFrame = None, eps_matrix: pd.DataFrame = None):
    """
    Renders Tab 1: Asset Dashboard (Pure Analysis Layer).
    """
    
    # Use pre-computed rolling returns if provided, else calculate locally
    if rolling_returns is not None:
        full_rolling_returns = rolling_returns
    else:
        with st.spinner("Calculating Rolling Returns Matrix…"):
            full_rolling_returns = math_engine.calculate_rolling_returns(window_days)
    
    # Slice results for display based on Historical Lookback
    start_lookback, end_lookback = lookback_range
    rolling_returns = full_rolling_returns.loc[start_lookback:end_lookback]
    sliced_master_matrix = master_matrix.loc[start_lookback:end_lookback]

    # Slice to primary asset
    primary_returns = rolling_returns[primary_ticker].dropna()

    if primary_returns.empty:
        st.warning(
            f"⚠️ **Insufficient Data:** Cannot calculate {window_label} rolling returns for "
            f"**{universe[primary_ticker]['name']}**. "
            f"The asset may have too short a history for this window. "
            f"Try a shorter window or check the Help tab."
        )
        return

    cagr_matrix = math_engine.calculate_cagr(rolling_returns, window_days)
    percentiles = math_engine.get_cross_sectional_percentiles(rolling_returns)

    # Current KPIs
    current_ret = primary_returns.iloc[-1]
    current_cagr = cagr_matrix[primary_ticker].dropna().iloc[-1]
    win_rate = (primary_returns > 0).mean()
    
    # Historical expectation metrics
    avg_ret = primary_returns.mean()
    p50_ret = primary_returns.median()
    p30_ret = np.percentile(primary_returns, 30)
    
    # New Metrics: Mean Annualized CAGR and Worst Return (P0)
    years = window_days / TRADING_DAYS_PER_YEAR
    mean_cagr = (1 + avg_ret) ** (1 / years) - 1
    worst_ret = primary_returns.min()

    # New Metric: Max Drawdown (Slices series based on lookback)
    p_prices = sliced_master_matrix[primary_ticker].dropna()
    p_cummax = p_prices.cummax()
    p_drawdown = (p_prices - p_cummax) / p_cummax
    max_dd = p_drawdown.min()

    # Universe rank
    current_row = rolling_returns.iloc[-1].dropna()
    rank_pos = (current_row > current_ret).sum() + 1
    rank_str = f"#{rank_pos} of {len(current_row)}"

    # Metadata header
    meta = universe[primary_ticker]
    asset_class_badge = f"`{meta.get('class', 'Unknown')}`"
    asset_type_label = "Total Return (ETF)" if meta.get('type') == 'TR' else "Price Return (Index)"
    st.markdown(
        f"**{primary_ticker}** &nbsp;|&nbsp; {meta['name']} &nbsp;|&nbsp; "
        f"{asset_class_badge} &nbsp;|&nbsp; {asset_type_label} &nbsp;|&nbsp; "
        f"Inception: `{meta.get('inception', 'Unknown')}`",
        unsafe_allow_html=True
    )

    # TR/PR mismatch warning
    primary_type = meta.get('type')
    if any(universe[b].get('type') != primary_type for b in benchmarks):
        st.warning(
            "⚠️ **TR/PR Mismatch:** You are comparing a Total Return ETF with a Price Return Index "
            "(or vice-versa). The ETF will appear to outperform due to dividends. "
            "See the **Help & Guide** tab → *TR vs PR* section for details."
        )

    # Calculate Bubble Z-Score over lifetime
    full_prices = master_matrix[primary_ticker].dropna()
    full_bubble_z_series = calculate_bubble_z_score(full_prices)
    bubble_z_series = full_bubble_z_series.loc[start_lookback:end_lookback]
    
    latest_bubble_z = full_bubble_z_series.iloc[-1] if not full_bubble_z_series.empty else np.nan
    bubble_status = classify_bubble_status(latest_bubble_z)

    # ── Row 1: KPIs ──────────────────────────────────────────
    render_kpi_dashboard(
        meta['name'], current_ret, current_cagr, win_rate, rank_str,
        avg_ret, mean_cagr, p50_ret, p30_ret, max_dd, worst_ret,
        window_days=window_days
    )

    # ── Bubble Risk Indicator strip ──────────────────────────
    st.markdown("#### 🫧 Macro Bubble Risk")
    b_col1, b_col2 = st.columns([1, 3])
    with b_col1:
        st.metric(
            label="Bubble Z-Score", 
            value=f"{latest_bubble_z:.2f} σ" if not pd.isna(latest_bubble_z) else "N/A",
            delta=None
        )
    with b_col2:
        getattr(st, bubble_status["badge"])(
            f"**Bubble Status: {bubble_status['status']}** ({bubble_status['emoji']})\n\n"
            f"Measures historical price divergence from its 3-year trendline, standardized against its lifetime history."
        )

    # ── Methodology Drill-down ──────────────────────────────
    render_methodology_drilldown()

    # ── Contextual KPI help strip ────────────────────────────
    with st.expander("ℹ️ What do these KPIs mean?", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.info("**Rolling Return** — The raw return if you had purchased this asset exactly N days ago and held until today.")
        c2.info("**CAGR** — Same return, annualized so all windows are comparable. Formula: `(1 + R)^(252/N) - 1`")
        c3.info("**Win Rate** — % of all historical N-day windows that were profitable. >70% = very consistent.")
        c4.info("**Universe Rank** — Today's rank among all 21 assets. #1 = highest rolling return right now.")

    # ── Row 2: Time-Series Chart ─────────────────────────────
    b_dict = {b: rolling_returns[b].dropna() for b in benchmarks}
    
    st.markdown("#### Chart Shading Controls")
    shading_type = st.radio(
        "Select Chart Shading Mode",
        ["Volatility & Trend Regimes", "Bubble Risk Regimes", "None"],
        horizontal=True,
        index=0,
        key="shading_mode"
    )

    if shading_type == "Volatility & Trend Regimes":
        daily_rets = sliced_master_matrix[primary_ticker].dropna().pct_change().dropna()
        vol_series = daily_rets.rolling(window=min(252, len(daily_rets)//2)).std() * np.sqrt(252)
        regime_series = classify_regime_2d(vol_series, sliced_master_matrix[primary_ticker].dropna(), window=min(252, len(daily_rets)//2))
    elif shading_type == "Bubble Risk Regimes":
        regime_series = classify_bubble_series(bubble_z_series)
    else:
        regime_series = None
    
    render_timeseries_chart(primary_returns, b_dict, percentiles, title=f"Rolling {window_label} Returns vs Universe",
                            regime_series=regime_series)

    # Chart help strip
    with st.expander("ℹ️ How to read this chart", expanded=False):
        st.markdown("""
        - 🔵 **Blue line** — Rolling return of the selected primary asset at each historical date.
        - ⬜ **Grey band** — 25th to 75th percentile of the entire 21-asset universe. This is the "normal zone."
        - ➖ **Dashed grey line** — Universe Median (50th percentile).
        - 🔴 **Red lines** — Benchmark overlays selected in the sidebar.
        
        **Tactical signals:**
        - Blue line **above** the grey band → Asset is in the **top 25%** of the universe *(momentum / extended)*
        - Blue line **inside** the grey band → Asset is **average** *(neutral)*
        - Blue line **below** the grey band → Asset is in the **bottom 25%** *(value / mean-reversion candidate)*
        """)

    # ── Fundamentals & Valuations Chart ──────────────────────
    if pe_matrix is not None and primary_ticker in pe_matrix.columns:
        pe_series = pe_matrix[primary_ticker].loc[start_lookback:end_lookback].dropna()
        if not pe_series.empty:
            st.markdown("---")
            st.markdown("### 📊 Valuation & Earnings Analysis")
            if primary_ticker in VAL_CROSS_CHECK_MAP:
                st.info(
                    f"ℹ️ **Data Verification:** Model-estimated from historical benchmarks and verified/validated daily "
                    f"against live ETF market references (reference: **{VAL_CROSS_CHECK_MAP[primary_ticker]}**)."
                )
            else:
                st.warning(
                    f"⚠️ **Data Disclaimer:** Purely model-estimated using historical index benchmarks. "
                    f"No live ETF reference exists for validation (e.g. Nikkei 225). Treat as directional/approximate only."
                )
            st.markdown("Long-term P/E ratio expansion/contraction mapped against corporate earnings momentum.")
            
            eps_series = eps_matrix[primary_ticker].dropna()
            price_series = master_matrix[primary_ticker].dropna()
            eps_growth_series = eps_series.pct_change(252).loc[start_lookback:end_lookback].dropna()
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig_fund = make_subplots(specs=[[{"secondary_y": True}]])
            
            # P/E Ratio (Left Y-axis)
            fig_fund.add_trace(
                go.Scatter(
                    x=pe_series.index, y=pe_series.values,
                    name="P/E Ratio (Left)", line=dict(color="#FF4B4B", width=2.5)
                ),
                secondary_y=False
            )
            
            # YoY EPS Growth (Right Y-axis)
            if not eps_growth_series.empty:
                fig_fund.add_trace(
                    go.Scatter(
                        x=eps_growth_series.index, y=eps_growth_series.values,
                        name="YoY EPS Growth (Right)", line=dict(color="#00CC96", width=2, dash="dash")
                    ),
                    secondary_y=True
                )
                
            fig_fund.update_layout(
                title=f"P/E Ratio & YoY EPS Growth for {universe[primary_ticker]['name']}",
                xaxis_title="Date",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            fig_fund.update_yaxes(title_text="P/E Ratio", secondary_y=False)
            fig_fund.update_yaxes(title_text="YoY EPS Growth", tickformat=".1%", secondary_y=True)
            
            # Sub-Tab selectors for fundamental metrics
            tab_val, tab_roll = st.tabs(["🏷️ P/E & YoY EPS Growth", "📈 Price vs. EPS Rolling Returns"])
            
            with tab_val:
                st.plotly_chart(fig_fund, use_container_width=True)
                
            with tab_roll:
                eps_rolling_return = eps_series.pct_change(window_days).loc[start_lookback:end_lookback].dropna()
                price_rolling_return = price_series.pct_change(window_days).loc[start_lookback:end_lookback].dropna()
                
                # Additive Return Decomposition
                # Price Return = EPS Return + Speculative Return (Multiple Expansion)
                speculative_contrib = price_rolling_return - eps_rolling_return
                
                # Calculate historical quantiles of speculative contribution for guidelines
                spec_p10 = speculative_contrib.quantile(0.10) if not speculative_contrib.empty else np.nan
                spec_p50 = speculative_contrib.quantile(0.50) if not speculative_contrib.empty else np.nan
                spec_p90 = speculative_contrib.quantile(0.90) if not speculative_contrib.empty else np.nan
                
                fig_roll = go.Figure()
                
                # Total Price Return
                fig_roll.add_trace(
                    go.Scatter(
                        x=price_rolling_return.index, y=price_rolling_return.values,
                        name="Total Price Return", line=dict(color="#1F77B4", width=3)
                    )
                )
                
                # Fundamental Contribution (EPS Growth)
                fig_roll.add_trace(
                    go.Scatter(
                        x=eps_rolling_return.index, y=eps_rolling_return.values,
                        name="Fundamental Return (EPS Growth)", line=dict(color="#2CA02C", width=2.5)
                    )
                )
                
                # Speculative Contribution (Multiple Expansion)
                fig_roll.add_trace(
                    go.Scatter(
                        x=speculative_contrib.index, y=speculative_contrib.values,
                        name="Speculative Return (Multiple Expansion)", line=dict(color="#FF7F0E", width=2.5)
                    )
                )
                
                # Horizontal Percentile Guidelines for Speculative Contribution
                if not speculative_contrib.empty:
                    dates_line = [speculative_contrib.index[0], speculative_contrib.index[-1]]
                    fig_roll.add_trace(
                        go.Scatter(
                            x=dates_line, y=[spec_p90, spec_p90],
                            name="Speculative 90th pctl (Overextended)", line=dict(color="#D62728", width=1.2, dash="dot"),
                            mode="lines"
                        )
                    )
                    fig_roll.add_trace(
                        go.Scatter(
                            x=dates_line, y=[spec_p50, spec_p50],
                            name="Speculative Median", line=dict(color="#7F7F7F", width=1.2, dash="dot"),
                            mode="lines"
                        )
                    )
                    fig_roll.add_trace(
                        go.Scatter(
                            x=dates_line, y=[spec_p10, spec_p10],
                            name="Speculative 10th pctl (Discounted)", line=dict(color="#9467BD", width=1.2, dash="dot"),
                            mode="lines"
                        )
                    )
                
                fig_roll.update_layout(
                    title=f"Price vs EPS Rolling {window_label} Return Decomposition for {universe[primary_ticker]['name']}",
                    xaxis_title="Date",
                    yaxis_title="Rolling Return / Contribution",
                    yaxis_tickformat=".1%",
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_roll, use_container_width=True)
                
                with st.expander("ℹ️ How to interpret the Return Decomposition", expanded=False):
                    st.markdown("""
                    This chart decomposes the **Total Price Return** into its fundamental and speculative drivers on a single unified axis:
                    
                    *   🔵 **Total Price Return:** The final net return of the asset over the selected window.
                    *   🟢 **Fundamental Return (EPS Growth):** The return contribution from corporate earnings growth. Higher and positive is healthier.
                    *   🟠 **Speculative Return (Multiple Expansion):** The return contribution driven strictly by multiple expansion (investors paying more per unit of earnings).
                    
                    **Tactical Valuation Anchors:**
                    *   🔴 **Speculative 90th percentile (Overextended):** Historically, when multiple expansion exceeds this line, the asset is extremely expensive relative to its own corporate earnings. Future return expectations are low, and drawdown risk is high.
                    *   🟣 **Speculative 10th percentile (Discounted):** Multiple contraction has reached historical extremes. The asset is fundamentally cheap, presenting strong potential mean-reversion value.
                    """)

    # ── Row 3: Distribution & Risk Analytics ─────────────────
    col_dist, col_stats = st.columns(2)
    with col_dist:
        render_distribution_matrix(primary_returns)
        with st.expander("ℹ️ Reading the distribution", expanded=False):
            st.markdown("""
            This histogram shows the **frequency of all historical rolling returns**.
            - A **right-skewed** distribution means large positive returns happen more often (e.g., Gold).
            - A **left-skewed** distribution means crash risk is higher than average.
            - If today's return is in the **far right tail**, the asset is at a historical peak — consider elevated reversion risk.
            """)

    with col_stats:
        st.markdown("### 🔬 Risk Analytics")
        import importlib.util
        if importlib.util.find_spec("vectorbt"):
            try:
                # Fix: Use daily returns for statistics, not rolling returns
                daily_returns = master_matrix[primary_ticker].dropna().pct_change().dropna()
                # Calculate stats with daily frequency to enable Sharpe/Sortino
                stats = _cached_vbt_stats(daily_returns)
                
                # --- Humanization & Alignment ---
                # Select and rename metrics for clarity and documentation alignment
                keep_metrics = {
                    'Start': 'Start Date',
                    'End': 'End Date',
                    'Total Return [%]': 'Total Return (%)',
                    'Max Drawdown [%]': 'Worst Historical Loss (%)',
                    'Sharpe Ratio': 'Sharpe Ratio (Risk-Adj)',
                    'Sortino Ratio': 'Sortino Ratio (Downside-Adj)',
                    'Tail Ratio': 'Tail Ratio (Upside/Downside)',
                    'Kurtosis': 'Kurtosis (Crash Probability)'
                }
                
                # Filter existing metrics
                available_metrics = [m for m in keep_metrics.keys() if m in stats.index]
                stats_filtered = stats[available_metrics].copy()
                
                # Format values for "Normal Investors"
                stats_display = {}
                for old_key, new_label in keep_metrics.items():
                    if old_key not in stats_filtered:
                        continue
                    val = stats_filtered[old_key]
                    
                    if 'Date' in new_label or old_key in ['Start', 'End']:
                        # Format date to YYYY-MM-DD
                        stats_display[new_label] = pd.to_datetime(val).strftime('%Y-%m-%d')
                    elif '(%)' in new_label:
                        # Round percentages to 2 decimal places
                        stats_display[new_label] = f"{val:.2f}%"
                    else:
                        # Round ratios to 2 decimal places
                        stats_display[new_label] = f"{val:.2f}"
                
                # Render as a clean table
                st.table(pd.Series(stats_display, name="Value"))
            except Exception as e:
                st.info(f"ℹ️ Detailed statistics require a longer history.")
                # st.write(e) # Debugging
        else:
            st.markdown("Detailed stats requires vectorbt.")

        with st.expander("ℹ️ Key risk metrics explained", expanded=False):
            st.markdown("""
            | Metric | Good level | Interpretation |
            |---|---|---|
            | **Max Drawdown** | < 30% | Worst peak-to-trough loss in history |
            | **Sharpe Ratio** | > 1.0 | Risk-adjusted return; >2 is excellent |
            | **Sortino Ratio** | > 1.5 | Like Sharpe, but penalises only downside volatility |
            | **Tail Ratio** | > 1.0 | Upside tail larger than downside tail |
            | **Kurtosis** | < 4 | >3 means fatter crash tails than a normal distribution |
            """)
