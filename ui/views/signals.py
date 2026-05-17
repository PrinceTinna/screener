import streamlit as st
import pandas as pd
import numpy as np
from core.state_math import calculate_z_score, calculate_trend, calculate_ols_slope, classify_regime_2d
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
                                 regime: str, prev_state: str = "Neutral") -> dict:
    """
    State-machine logic for signal generation with Hysteresis.
    Uses different entry/exit thresholds to prevent flickering.
    """
    is_extreme_breakdown = regime == "Extreme Breakdown"
    is_high_vol = regime in ("Panic", "Recovery", "Extreme Breakdown")

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
            badge = "warning" if is_high_vol else "success"
            label = "🟡 Weak Momentum (Whipsaw Risk)" if is_high_vol else "🟢 Momentum"
            return {
                "signal": label,
                "badge": badge,
                "confidence": "Holding (Hysteresis)",
                "detail": f"Z={z_score:.2f}, Rank={rank_pct*100:.0f}th, Slope={slope:.4f}",
                "state": "Momentum"
            }
    else:
        # Not in momentum — use ENTRY thresholds (tighter)
        if (rank_pct > MOMENTUM_ENTRY["rank_pct"]
                and z_score > MOMENTUM_ENTRY["z_score"]
                and slope >= 0):
            badge = "warning" if is_high_vol else "success"
            label = "🟡 Weak Momentum (Whipsaw Risk)" if is_high_vol else "🟢 Momentum"
            return {
                "signal": label,
                "badge": badge,
                "confidence": "New Entry",
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
           rolling_returns_matrix: pd.DataFrame = None, universe: dict = None):
    """
    Renders Tab 2: Signals (Execution Layer) with Regime-Aware Hysteresis.
    """
    st.markdown("## 🎯 Regime-Aware Signal Engine")
    st.markdown("Production-grade directives conditioned by volatility regime and trend direction.")

    if primary_ticker not in master_matrix.columns:
        st.error("Primary ticker data not available.")
        return

    price_series = master_matrix[primary_ticker].dropna()
    if len(price_series) < 252:
        st.warning("⚠️ Insufficient historical data for signal generation (need 252+ days).")
        return

    rolling_series = (rolling_returns_matrix[primary_ticker].dropna()
                      if rolling_returns_matrix is not None else None)

    # ── Core Calculations ─────────────────────────────────────────────
    with st.spinner("Computing Regime & Signal State..."):
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

        # Segment Rank (cross-sectional)
        if rolling_returns_matrix is not None:
            latest_returns = rolling_returns_matrix.iloc[-1].dropna()
            current_return = latest_returns.get(primary_ticker, 0.0)
            rank_pct = (latest_returns < current_return).sum() / max(len(latest_returns), 1)
        else:
            rank_pct = 0.5  # Default to median if no cross-sectional data

        # Hysteresis: Retrieve previous state from session
        prev_state = st.session_state.get(f"signal_state_{primary_ticker}", "Neutral")

        # Evaluate signal
        signal = _evaluate_hysteresis_signal(
            z_score=current_z,
            rank_pct=rank_pct,
            slope=current_slope,
            regime=current_regime,
            prev_state=prev_state
        )

        # Persist state for hysteresis
        st.session_state[f"signal_state_{primary_ticker}"] = signal["state"]

        # Arbitration Engine (existing)
        engine = ArbitrationEngine(price_series, rolling_series)
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

    # Signal banner
    getattr(st, signal["badge"])(f"### {signal['signal']}")

    s1, s2, s3 = st.columns(3)
    s1.metric("Z-Score", f"{current_z:.2f} σ")
    s2.metric("Confidence", signal["confidence"])
    s3.metric("State Machine", signal["state"])

    st.caption(f"**Signal Detail:** {signal['detail']}")

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
    # SECTION D: INVERSION TRANSPARENCY MATRIX
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("### 🔹 Inversion Transparency Matrix")
    st.info(f"**Veto Analysis:** {inversion_matrix['veto_printout']}")

    with st.expander(f"View Historical Events ({inversion_matrix['events_found']} Found)", expanded=False):
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
