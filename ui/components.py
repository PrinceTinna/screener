import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_kpi_dashboard(asset_name: str, current_return: float, cagr: float, win_rate: float, rank: str, 
                         mean_ret: float, mean_cagr: float, p50: float, p30: float, 
                         max_dd: float, worst_ret: float, window_days: int = None):
    """Renders the top row of KPI cards and historical expectations."""
    st.markdown(f"### 📊 Insight: {asset_name}")
    
    is_1y = (window_days == 252)
    cagr_help = "💡 Annualized CAGR is mathematically identical to simple return at a 1-year holding window." if is_1y else "Annualized Compound Annual Growth Rate."
    mean_cagr_help = "💡 Mean Annualized CAGR is mathematically identical to rolling mean return at a 1-year holding window." if is_1y else "Annualized rolling mean return."
    
    # Row 1: Snapshot Metrics
    st.markdown("#### Real-Time Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Current Return", value=f"{current_return*100:.2f}%")
    with col2:
        st.metric(label="Annualized CAGR", value=f"{cagr*100:.2f}%", help=cagr_help)
    with col3:
        st.metric(label="Historical Win Rate", value=f"{win_rate*100:.1f}%")
    with col4:
        st.metric(label="Universe Rank", value=rank)

    # Row 2: Expectation Metrics
    st.markdown("#### Historical Expectations (Holding Period Strategy)")
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric(label="Mean Rolling Return", value=f"{mean_ret*100:.2f}%")
    with e2:
        st.metric(label="Mean Annualized CAGR", value=f"{mean_cagr*100:.2f}%", help=mean_cagr_help)
    with e3:
        st.metric(label="P50 (Median) Return", value=f"{p50*100:.2f}%")
    with e4:
        st.metric(label="P30 (Conservative) Return", value=f"{p30*100:.2f}%")

    # Row 3: Risk Metrics
    st.markdown("#### Historical Risk Profile")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric(label="Max Drawdown (Full)", value=f"{max_dd*100:.2f}%", delta=None)
    with r2:
        st.metric(label="Worst Rolling Period", value=f"{worst_ret*100:.2f}%", delta=None)
    with r3:
        # Visual delta from current to mean
        delta = (current_return - mean_ret) * 100
        st.metric(label="Current vs Mean Delta", value=f"{delta:.2f}%", delta=f"{delta:.2f}%", delta_color="inverse")

def render_methodology_drilldown():
    """Renders a collapsible section explaining metric calculations."""
    with st.expander("🔍 Methodology Drill-down: How are these calculated?", expanded=False):
        st.markdown("""
        ### 📐 Mathematical Definitions
        
        #### 1. Current Return
        Calculated as the simple percentage change from the price **N days ago** to today.
        - **Formula:** `(Price_today / Price_{today - N}) - 1`
        
        #### 2. Mean Rolling Return
        The average of every possible N-day return in the asset's history. It represents the "Expected Value" if you enter on a random day.
        - **Formula:** `Average(All Rolling Returns for window N)`
        
        #### 3. Annualized CAGR (Mean)
        Converts the Mean Rolling Return into a standard yearly rate.
        - **Formula:** `(1 + Mean_Rolling_Return)^(252 / N) - 1`
        
        #### 4. P50 Median & P30 Conservative
        - **P50 (Median):** The 50th percentile. Half of history was better than this, half was worse.
        - **P30 (Conservative):** The 30th percentile. 70% of history was better than this level.
        
        #### 5. Max Drawdown (Full)
        The largest peak-to-trough decline recorded in the asset's **entire price index history**, regardless of your holding period.
        - **Formula:** `(Trough_Value - Peak_Value) / Peak_Value`
        
        #### 6. Worst Rolling Period (P0)
        The absolute lowest return ever recorded for the selected window N. This tells you the "Maximum Pain" you could have felt in exactly N days.
        """)

def render_timeseries_chart(primary_series: pd.Series, benchmarks: dict, percentiles: pd.DataFrame, title: str,
                            regime_series: pd.Series = None):
    """Renders the main line chart with benchmark overlays, percentile bands, and optional 2D regime shading."""
    fig = go.Figure()

    # ── Regime Background Shading ────────────────────────────────────
    if regime_series is not None and not regime_series.empty:
        regime_color_map = {
            "Trending Bull": "rgba(0, 200, 83, 0.08)",
            "Neutral Bull": "rgba(0, 200, 83, 0.04)",
            "Recovery": "rgba(255, 193, 7, 0.08)",
            "Low-Vol Range": "rgba(158, 158, 158, 0.06)",
            "Neutral Bear": "rgba(255, 152, 0, 0.06)",
            "Panic": "rgba(244, 67, 54, 0.10)",
            "Extreme Breakdown": "rgba(183, 28, 28, 0.15)",
            "Unknown": "rgba(0, 0, 0, 0)",
            "Normal": "rgba(0, 200, 83, 0.03)",
            "Extended": "rgba(255, 193, 7, 0.05)",
            "2-Sigma Bubble": "rgba(255, 152, 0, 0.12)",
            "3-Sigma Superbubble": "rgba(244, 67, 54, 0.20)",
        }
        # Align regime_series to primary_series index
        aligned_regime = regime_series.reindex(primary_series.index, method='ffill')
        
        # Create contiguous regime blocks for efficient rendering
        if len(aligned_regime) > 0:
            shapes = []
            annotations = []
            current_regime = aligned_regime.iloc[0]
            block_start = aligned_regime.index[0]
            
            for i in range(1, len(aligned_regime)):
                if aligned_regime.iloc[i] != current_regime or i == len(aligned_regime) - 1:
                    block_end = aligned_regime.index[i]
                    color = regime_color_map.get(current_regime, "rgba(0,0,0,0)")
                    if color != "rgba(0, 0, 0, 0)":
                        shapes.append(dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=block_start,
                            x1=block_end,
                            y0=0,
                            y1=1,
                            fillcolor=color,
                            opacity=1.0,
                            layer="below",
                            line_width=0,
                        ))
                        if (block_end - block_start).days > 90:
                            annotations.append(dict(
                                x=block_start,
                                y=1.0,
                                xref="x",
                                yref="paper",
                                text=current_regime,
                                showarrow=False,
                                font=dict(size=8, color="grey"),
                                xanchor="left",
                                yanchor="top"
                            ))
                    current_regime = aligned_regime.iloc[i]
                    block_start = aligned_regime.index[i]
            
            fig.update_layout(shapes=shapes, annotations=annotations)

    # Percentile shading
    if not percentiles.empty:
        fig.add_trace(go.Scatter(
            x=percentiles.index, y=percentiles['P75'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=percentiles.index, y=percentiles['P25'],
            mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(128,128,128,0.2)',
            name='25th-75th Percentile Core', showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=percentiles.index, y=percentiles['Median'],
            mode='lines', line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash'),
            name='Universe Median'
        ))

    # Benchmarks
    for b_name, b_series in benchmarks.items():
        fig.add_trace(go.Scatter(
            x=b_series.index, y=b_series.values,
            mode='lines', line=dict(width=1, color='rgba(200, 100, 100, 0.6)'),
            name=f"Bench: {b_name}"
        ))

    # Primary Asset
    fig.add_trace(go.Scatter(
        x=primary_series.index, y=primary_series.values,
        mode='lines', line=dict(color='#007FFF', width=3),
        name=primary_series.name or "Primary Asset"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rolling Return",
        yaxis=dict(tickformat='.0%'),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_distribution_matrix(series: pd.Series):
    """Renders a KDE histogram representing return distributions."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=series.values,
        histnorm='probability density',
        marker_color='#007FFF',
        opacity=0.75,
        name='Distribution'
    ))
    
    # Highlight today's return
    today_ret = series.iloc[-1]
    fig.add_vline(
        x=today_ret, 
        line_width=3, 
        line_dash="dash", 
        line_color="#FF4B4B", 
        annotation_text=f"Today: {today_ret*100:.2f}%", 
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Return Distribution (Density)",
        xaxis_title="Return Bucket",
        yaxis_title="Density",
        xaxis=dict(tickformat='.0%'),
        template="plotly_dark",
        bargap=0.05
    )
    
    st.plotly_chart(fig, use_container_width=True)
