import streamlit as st
import pandas as pd
import numpy as np
from core.state_math import calculate_z_score, calculate_trend, calculate_ols_slope, classify_regime_2d, calculate_bubble_z_score, classify_bubble_status
from strategies.arbitration import ArbitrationEngine

# ── Hysteresis Thresholds ────────────────────────────────────────────────
# Different entry/exit thresholds prevent signal flickering
MOMENTUM_ENTRY = {"rank_pct": 0.75, "z_score": 0.5}
MOMENTUM_EXIT  = {"rank_pct": 0.60, "z_score": 0.0}
REVERSAL_ENTRY = {"rank_pct": 0.25, "z_score": -1.0}
REVERSAL_EXIT  = {"rank_pct": 0.40, "z_score": -0.5}

# ── Regime Color Map for UI ──────────────────────────────────────────────
REGIME_COLORS = {
    "Trending Bull": ("🟢", "success"),
    "Neutral Bull": ("🟢", "success"),
    "Recovery": ("🟡", "warning"),
    "Low-Vol Range": ("🟡", "warning"),
    "Neutral Bear": ("🟠", "warning"),
    "Panic": ("🔴", "error"),
    "Extreme Breakdown": ("⛔", "error"),
    "Unknown": ("⚪", "info"),
}

# ── Strategy Bias Map ────────────────────────────────────────────────────
REGIME_STRATEGY = {
    "Trending Bull": "Momentum strategies are prioritized. Ride the trend.",
    "Neutral Bull": "Momentum with moderate confidence. Watch for vol expansion.",
    "Recovery": "Volatility-adjusted momentum. Mean-reversion signals gaining strength.",
    "Low-Vol Range": "Range-bound market. Neither momentum nor reversal dominant.",
    "Neutral Bear": "Trend weakening. Reduce exposure; wait for confirmation.",
    "Panic": "Mean-reversion signals are prioritized. Look for capitulation bounces.",
    "Extreme Breakdown": "⛔ HARD VETO: All momentum signals DISABLED. Breakdown regime.",
    "Unknown": "Insufficient data for regime classification.",
}


def _evaluate_hysteresis_signal(z_score: float, rank_pct: float, slope: float,
                                 regime: str, prev_state: str = "Neutral", bubble_z: float = 0.0,
                                 pe_percentile: float = np.nan, is_data_gap: bool = False) -> dict:
    """
    State-machine logic for signal generation with Hysteresis.
    Uses different entry/exit thresholds to prevent flickering.
    """
    if is_data_gap:
        return {
            "signal": "🔴 High Uncertainty (Data Gap)",
            "badge": "error",
            "confidence": "Data Integrity Failure",
            "detail": "Severe data gaps (missing price bars > 5%) combined with extreme CAGR outlier (>150%).",
            "state": "High Uncertainty (Data Gap)"
        }

    is_extreme_breakdown = regime == "Extreme Breakdown"
    is_high_vol = regime in ("Panic", "Recovery", "Extreme Breakdown")
    is_overvalued = not pd.isna(pe_percentile) and pe_percentile >= 0.90

    # ── Hard Veto: Extreme Vol kills momentum entirely ─────────────────
    if is_extreme_breakdown:
        return {
            "signal": "⛔ VETO (Breakdown)",
            "badge": "error",
            "confidence": "Hard Veto Active",
            "detail": "Volatility exceeds 90th percentile. All momentum signals disabled.",
            "state": "Veto"
        }

    # ── Momentum Logic (with Hysteresis) ───────────────────────────────
    # Entry: Rank > 75th AND Z > 0.5 AND Slope >= 0
    # Exit:  Rank < 60th OR Z < 0
    if prev_state == "Momentum":
        # Already in momentum — use EXIT thresholds (wider)
        if rank_pct < MOMENTUM_EXIT["rank_pct"] or z_score < MOMENTUM_EXIT["z_score"]:
            pass  # Fall through to check other states
        else:
            is_bubble = not pd.isna(bubble_z) and bubble_z >= 2.0
            badge = "warning" if (is_high_vol or is_bubble or is_overvalued) else "success"
            if is_overvalued:
                label = "🟡 Weak Momentum (Overvaluation Risk)"
                confidence = "Holding (Overvalued PE >= 90th percentile)"
            elif is_bubble:
                label = "🟡 Weak Momentum (Bubble Risk)"
                confidence = "Holding (Bubble Risk Active)"
            else:
                label = "🟡 Weak Momentum (Whipsaw Risk)" if is_high_vol else "🟢 Momentum"
                confidence = "Holding (Hysteresis)"
            return {
                "signal": label,
                "badge": badge,
                "confidence": confidence,
                "detail": f"Z={z_score:.2f}, Rank={rank_pct*100:.0f}th, Slope={slope:.4f}",
                "state": "Momentum"
            }
    else:
        # Not in momentum — use ENTRY thresholds (tighter)
        if (rank_pct > MOMENTUM_ENTRY["rank_pct"]
                and z_score > MOMENTUM_ENTRY["z_score"]
                and slope >= 0):
            is_bubble = not pd.isna(bubble_z) and bubble_z >= 2.0
            badge = "warning" if (is_high_vol or is_bubble or is_overvalued) else "success"
            if is_overvalued:
                label = "🟡 Weak Momentum (Overvaluation Risk)"
                confidence = "New Entry (Overvalued PE >= 90th percentile)"
            elif is_bubble:
                label = "🟡 Weak Momentum (Bubble Risk)"
                confidence = "New Entry (Bubble Risk Active)"
            else:
                label = "🟡 Weak Momentum (Whipsaw Risk)" if is_high_vol else "🟢 Momentum"
                confidence = "New Entry"
            return {
                "signal": label,
                "badge": badge,
                "confidence": confidence,
                "detail": f"Z={z_score:.2f}, Rank={rank_pct*100:.0f}th, Slope={slope:.4f}",
                "state": "Momentum"
            }

    # ── Reversal Logic (Falling Knife Guard) ───────────────────────────
    # Entry: Rank < 25th AND Z < -1.0 AND Slope > 0
    # Exit:  Rank > 40th OR Z > -0.5
    if prev_state == "Reversal":
        if rank_pct > REVERSAL_EXIT["rank_pct"] or z_score > REVERSAL_EXIT["z_score"]:
            pass  # Fall through to neutral
        else:
            badge = "success" if is_high_vol else "error"
            label = "🟢 Strong Reversal" if is_high_vol else "🔴 Reversal Zone"
            return {
                "signal": label,
                "badge": badge,
                "confidence": "Holding (Hysteresis)",
                "detail": f"Z={z_score:.2f}, Rank={rank_pct*100:.0f}th, Slope={slope:.4f}",
                "state": "Reversal"
            }
    else:
        if (rank_pct < REVERSAL_ENTRY["rank_pct"]
                and z_score < REVERSAL_ENTRY["z_score"]
                and slope > 0):
            badge = "success" if is_high_vol else "error"
            label = "🟢 Strong Reversal" if is_high_vol else "🔴 Reversal Zone"
            return {
                "signal": label,
                "badge": badge,
                "confidence": "New Entry",
                "detail": f"Z={z_score:.2f}, Rank={rank_pct*100:.0f}th, Slope={slope:.4f}",
                "state": "Reversal"
            }

    # ── Neutral / Underperforming ──────────────────────────────────────
    return {
        "signal": "⚪ Neutral",
        "badge": "info",
        "confidence": "No Signal",
        "detail": f"Z={z_score:.2f}, Rank={rank_pct*100:.0f}th, Slope={slope:.4f}",
        "state": "Neutral"
    }


def render(master_matrix: pd.DataFrame, primary_ticker: str,
           rolling_returns_matrix: pd.DataFrame = None, universe: dict = None,
           pe_matrix: pd.DataFrame = None, eps_matrix: pd.DataFrame = None):
    """
    Renders Tab 2: Signals (Execution Layer) with Regime-Aware Hysteresis.
    """
    st.markdown("## 🎯 Regime-Aware Signal Engine")
    st.markdown("Production-grade directives conditioned by volatility regime and trend direction.")

    if universe is not None and primary_ticker in universe:
        meta = universe[primary_ticker]
        asset_class_badge = f"`{meta.get('class', 'Unknown')}`"
        asset_type_label = "Total Return (ETF)" if meta.get('type') == 'TR' else "Price Return (Index)"
        st.markdown(
            f"**{primary_ticker}** &nbsp;|&nbsp; {meta['name']} &nbsp;|&nbsp; "
            f"{asset_class_badge} &nbsp;|&nbsp; {asset_type_label} &nbsp;|&nbsp; "
            f"Inception: `{meta.get('inception', 'Unknown')}`",
            unsafe_allow_html=True
        )
        st.markdown("---")

    if primary_ticker not in master_matrix.columns:
        st.error("Primary ticker data not available.")
        return

    price_series = master_matrix[primary_ticker].dropna()
    total_history_days = len(price_series)

    rolling_series = (rolling_returns_matrix[primary_ticker].dropna()
                      if rolling_returns_matrix is not None else None)

    # Helper function for Check D: Cross-Sectional Sync Guard
    def _apply_cross_sectional_sync_guard(returns_series: pd.Series, target_date: pd.Timestamp) -> pd.Series:
        from config.settings import CACHE_DIR
        clean_series = returns_series.copy()
        for tkr in clean_series.index:
            raw_path = CACHE_DIR / f"{tkr}_raw.parquet"
            if raw_path.exists():
                try:
                    last_raw_date = pd.to_datetime(pd.read_parquet(raw_path, columns=[]).index[-1])
                    if (target_date - last_raw_date).days > 5:
                        clean_series[tkr] = np.nan
                except Exception:
                    clean_series[tkr] = np.nan
        return clean_series

    # ── Core Calculations ─────────────────────────────────────────────
    with st.spinner("Computing Regime & Signal State..."):
        # Default fallback values for short histories
        current_z = 0.0
        current_slope = 0.0
        current_vol = 0.0
        current_regime = "Unknown"
        rank_pct = 0.5
        current_pe = np.nan
        pe_pct = np.nan
        current_bubble_z = np.nan
        bubble_status = {
            "status": "Insufficient History",
            "badge": "warning",
            "emoji": "🟡",
            "label": "🟡 Insufficient History"
        }
        is_data_gap = False

        if total_history_days >= 252:
            # Z-Score
            z_series = calculate_z_score(
                rolling_series if rolling_series is not None else price_series.pct_change(252).dropna(),
                window=252
            )
            current_z = z_series.iloc[-1] if len(z_series.dropna()) > 0 else 0.0

            # OLS Slope (10-day regression on rolling returns)
            slope_input = rolling_series if rolling_series is not None else price_series.pct_change(252).dropna()
            slope_series = calculate_ols_slope(slope_input, window=10)
            current_slope = slope_series.iloc[-1] if len(slope_series.dropna()) > 0 else 0.0

            # Volatility (rolling 252-day annualized)
            daily_returns = price_series.pct_change().dropna()
            vol_series = daily_returns.rolling(window=252).std() * np.sqrt(252)
            current_vol = vol_series.iloc[-1] if len(vol_series.dropna()) > 0 else 0.0

            # 2D Regime Classification
            regime_series = classify_regime_2d(vol_series, price_series, window=252)
            current_regime = regime_series.iloc[-1] if len(regime_series) > 0 else "Unknown"

            # Check D: Cross-Sectional Sync Guard
            if rolling_returns_matrix is not None and not rolling_returns_matrix.empty:
                sync_returns = _apply_cross_sectional_sync_guard(rolling_returns_matrix.iloc[-1], rolling_returns_matrix.index[-1])
                latest_returns = sync_returns.dropna()
                current_return = latest_returns.get(primary_ticker, 0.0)
                rank_pct = (latest_returns < current_return).sum() / max(len(latest_returns), 1)

            # Get PE and calculate its historical percentile
            if pe_matrix is not None and primary_ticker in pe_matrix.columns:
                ticker_pe = pe_matrix[primary_ticker].dropna()
                if not ticker_pe.empty:
                    current_pe = ticker_pe.iloc[-1]
                    if len(ticker_pe) > 1:
                        pe_pct = (ticker_pe < current_pe).sum() / len(ticker_pe)

            # Check B: Bubble Detection Sample-Size Guard
            if total_history_days >= 1000:
                bubble_z_series = calculate_bubble_z_score(price_series)
                current_bubble_z = bubble_z_series.iloc[-1] if not bubble_z_series.empty else np.nan
                bubble_status = classify_bubble_status(current_bubble_z)

            # Check C: Fat-Tail Outlier CAGR Cap
            if rolling_series is not None and not rolling_series.empty:
                cagr = rolling_series.iloc[-1]
                if cagr > 1.50:
                    price_slice = price_series.tail(252)
                    missing_pct = price_slice.isna().sum() / 252
                    if missing_pct > 0.05:
                        is_data_gap = True

        # Hysteresis: Retrieve previous state from session
        prev_state = st.session_state.get(f"signal_state_{primary_ticker}", "Neutral")

        # Evaluate signal
        if total_history_days < 252:
            signal = {
                "signal": "🟡 Insufficient History",
                "badge": "warning",
                "confidence": "Insufficient History",
                "detail": "Asset must have at least 252 trading days (~1 year) of continuous data before any momentum indicator or signal is computed.",
                "state": "Insufficient History"
            }
        else:
            signal = _evaluate_hysteresis_signal(
                z_score=current_z,
                rank_pct=rank_pct,
                slope=current_slope,
                regime=current_regime,
                prev_state=prev_state,
                bubble_z=current_bubble_z,
                pe_percentile=pe_pct,
                is_data_gap=is_data_gap
            )

        # Persist state for hysteresis
        st.session_state[f"signal_state_{primary_ticker}"] = signal["state"]

        # Arbitration Engine (existing)
        engine = ArbitrationEngine(price_series, rolling_series if total_history_days >= 252 else None)
        action_grid = engine.generate_action_grid(current_z, price_series.iloc[-1], price_series.max())
        inversion_matrix = engine.run_inversion_matrix()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION A: REGIME STATUS PANEL
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("### 🔹 Current Market Regime")

    regime_icon, regime_badge = REGIME_COLORS.get(current_regime, ("⚪", "info"))
    regime_strategy = REGIME_STRATEGY.get(current_regime, "Unknown regime.")

    # Regime banner
    getattr(st, regime_badge)(
        f"### {regime_icon} {current_regime}\n\n"
        f"**Strategy Bias:** {regime_strategy}"
    )

    # Regime metrics
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Annualized Volatility", f"{current_vol*100:.1f}%")
    r2.metric("OLS Slope (β)", f"{current_slope:.4f}")
    trend_df = calculate_trend(price_series)
    trend_str = "Positive" if trend_df['is_positive_trend'].iloc[-1] else "Negative"
    r3.metric("Trend (50>200 SMA)", trend_str)
    r4.metric("Segment Rank", f"{rank_pct*100:.0f}th pctl")

    with st.expander("ℹ️ Understanding the 2D Regime Model", expanded=False):
        st.markdown("""
        The regime engine classifies markets along **two axes**:
        
        | Vol \\ Trend | ↑ Up | ↓ Down |
        |---|---|---|
        | **Low (<40th pctl)** | Trending Bull | Low-Vol Range |
        | **Neutral (40-75th)** | Neutral Bull | Neutral Bear |
        | **High (75-90th)** | Recovery | Panic |
        | **Extreme (>90th)** | ⛔ Breakdown | ⛔ Breakdown |
        
        - **Momentum** works best in Low Vol + Uptrend.
        - **Reversal** works best in High Vol + Downtrend.
        - **Extreme Breakdown** vetoes all momentum signals.
        """)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION B: REGIME-AWARE SIGNAL (HYSTERESIS)
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("### 🔹 Signal Engine (Hysteresis-Stabilized)")

    # Bubble Alert Banner
    if not pd.isna(current_bubble_z) and current_bubble_z >= 2.0:
        st.warning(
            f"### ⚠️ Statistical Bubble Warning ({bubble_status['emoji']} {bubble_status['status']})\n\n"
            f"The asset's Bubble Z-Score is **{current_bubble_z:.2f} σ**, indicating it is in extreme "
            f"statistical outlier territory. Historically, while momentum can persist (the 'Bull Trap'), "
            f"risk of severe drawdown is highly elevated. Systematic strategies suggest de-escalating exposure "
            f"rather than attempting to time the peak."
        )

    # Valuation Alert Banner
    if not pd.isna(pe_pct) and pe_pct >= 0.90:
        st.warning(
            f"### ⚠️ Valuation Warning (Top 10% PE Ratio)\n\n"
            f"The asset's current P/E ratio of **{current_pe:.2f}** is in the **{pe_pct*100:.1f}th percentile** "
            f"of its lifetime history. Investing in equity indices when valuations are in the top decile "
            f"historically yields compressed forward returns and elevated drawdown risk. "
            f"Allocations are de-escalated via the Valuation Gate."
        )

    # Veto Explain Callout
    if signal["state"] == "Veto":
        st.error(
            f"### ⛔ Hard Veto Active\n\n"
            f"**What this means:** Volatility is in the **top 10% of its historical baseline** (Annualized Volatility: **{current_vol*100:.1f}%**). "
            f"During extreme market stress, active momentum and trend indicators have low signal-to-noise ratios and are prone to whipsaw losses.\n\n"
            f"**Actionable Guidance:**\n"
            f"*   **Active Trading:** All active tactical momentum buys are temporarily suspended/disabled.\n"
            f"*   **Passive DCA / SIP:** Do **NOT** stop your systematic monthly SIP. Passive cost-averaging is mathematically designed to acquire units at lower prices. Keep standard allocations active."
        )

    # Signal banner
    getattr(st, signal["badge"])(f"### {signal['signal']}")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Z-Score", 
        f"{current_z:.2f} σ",
        help="📏 Statistical deviation of rolling N-day return from its 1-year mean/std. Entry: >0.5, Exit: <0.0."
    )
    s2.metric(
        "Bubble Z-Score", 
        f"{current_bubble_z:.2f} σ" if not pd.isna(current_bubble_z) else "N/A",
        help="🫧 Detrended price distance Z-score (3Y SMA baseline, lifetime history). Outlier: >=2.0, Superbubble: >=3.0."
    )
    s3.metric(
        "Confidence", 
        signal["confidence"],
        help="⚖️ State Machine reasoning and de-escalation/veto indicators."
    )
    s4.metric(
        "State Machine", 
        signal["state"],
        help="🔄 Core state tracking state (Momentum, Reversal, Neutral, Veto)."
    )

    st.caption(f"**Signal Detail:** {signal['detail']}")

    with st.expander("ℹ️ How today's signal was calculated (Formula & Rules)", expanded=False):
        st.markdown(rf"""
        Today's signal (`{signal['signal']}`) was generated by evaluating the following quantitative parameters:
        
        *   **Z-Score ({current_z:.2f} σ):** Standardized rolling return over the selected window. 
            *   *Formula:* $(R_{{t, w}} - \mu(R)) / \sigma(R)$ computed over a rolling 252-day window.
        *   **Segment Rank ({rank_pct*100:.0f}th percentile):** Cross-sectional performance ranking of the asset relative to peers in the same segment.
            *   *Condition:* Momentum requires rank > 75th percentile (Entry) or > 60th percentile (Exit).
        *   **OLS Slope ({current_slope:.4f}):** Noise-filtered trend directionality.
            *   *Formula:* Linear regression slope of returns over the last 10 days. Momentum entry requires slope $\ge 0$.
        *   **Market Volatility Regime ({current_regime}):** Under extreme volatility (>90th percentile of history), a hard veto blocks all Momentum signals.
        *   **Bubble Z-Score ({current_bubble_z:.2f} σ):** Detrended price distance.
            *   *Formula:* Price distance from its 3-year SMA, standardized over the asset's entire lifetime history. If $Z_{{bubble}} \ge 2.0$, momentum signals are downgraded.
        *   **Valuation Gate (P/E Percentile: {pe_pct*100:.1f}th if applicable):** Historical P/E ratio position.
            *   *Condition:* If current P/E is in the top 10% of history (percentile $\ge 90\%$), momentum signals are downgraded to Overvaluation Risk.
        
        **Current Execution State Machine:** `{signal['state']}`
        """)

    with st.expander("ℹ️ How Hysteresis Prevents Flickering", expanded=False):
        st.markdown(f"""
        **Hysteresis** uses different thresholds for **entering** and **exiting** a signal state:
        
        | Signal | Entry Condition | Exit Condition |
        |---|---|---|
        | **Momentum** | Rank >{MOMENTUM_ENTRY['rank_pct']*100:.0f}th AND Z>{MOMENTUM_ENTRY['z_score']} AND Slope≥0 | Rank <{MOMENTUM_EXIT['rank_pct']*100:.0f}th OR Z<{MOMENTUM_EXIT['z_score']} |
        | **Reversal** | Rank <{REVERSAL_ENTRY['rank_pct']*100:.0f}th AND Z<{REVERSAL_ENTRY['z_score']} AND Slope>0 | Rank >{REVERSAL_EXIT['rank_pct']*100:.0f}th OR Z>{REVERSAL_EXIT['z_score']} |
        
        This prevents daily noise from flipping signals back and forth.
        """)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION C: ACTION GRID (From existing ArbitrationEngine)
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("### 🔹 Action Grid (Capital Allocation)")

    col_sip, col_reserve = st.columns(2)

    with col_sip:
        st.markdown("#### Monthly SIP Flow")
        st.markdown(f"**Target Action:** `{action_grid['sip']['action']}`")
        st.markdown(f"**Recommended Multiplier:** `{action_grid['sip']['multiplier']}`")
        st.caption(f"**Rationale:** {action_grid['sip']['rationale']}")

    with col_reserve:
        st.markdown("#### Dry Powder Reserves")
        st.markdown(f"**Target Action:** `{action_grid['reserves']['action']}`")
        drawdown_pct = action_grid['reserves']['drawdown'] * 100
        st.markdown(f"**Current Drawdown:** `{drawdown_pct:.2f}%`")
        st.markdown(f"**Next Trigger:** `{action_grid['reserves']['next_trigger']}`")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION D: EMPIRICAL BACKTEST PROBABILITIES
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("### 📊 Empirical Backtest Probabilities")
    st.markdown(
        "*This panel scans the historical dataset to find all past instances where the asset experienced a similar signal state. "
        "It calculates the performance outcomes over the subsequent 3, 6, and 12 months to verify the historical reliability of today's signal.*"
    )
    st.info(f"**Backtest Analysis:** {inversion_matrix['veto_printout']}")

    with st.expander(f"View Historical Backtest Events ({inversion_matrix['events_found']} Found)", expanded=False):
        if inversion_matrix['events_found'] > 0:
            df_hist = inversion_matrix['history_table'].copy()
            for col in ['Next 3M Return', 'Next 6M Return']:
                if col in df_hist.columns:
                    df_hist[col] = df_hist[col].apply(
                        lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            if 'Max Drawdown (Next 1Y)' in df_hist.columns:
                df_hist['Max Drawdown (Next 1Y)'] = df_hist['Max Drawdown (Next 1Y)'].apply(
                    lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            if 'Z-Score' in df_hist.columns:
                df_hist['Z-Score'] = df_hist['Z-Score'].apply(lambda x: f"{x:.2f}")
            st.table(df_hist)
        else:
            st.info("No comparable historical events found for the current threshold.")
