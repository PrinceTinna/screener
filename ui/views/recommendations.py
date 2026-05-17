import streamlit as st
import pandas as pd
from strategies.arbitration import ArbitrationEngine

def render(master_matrix: pd.DataFrame, primary_ticker: str, rolling_returns_matrix: pd.DataFrame = None):
    """
    Renders Tab 2: Recommendations (Execution Layer).
    """
    st.markdown("## 🎯 Execution Strategy")
    st.markdown("Operationalizing historical data into actionable directives.")
    
    if primary_ticker not in master_matrix.columns:
        st.error("Primary ticker data not available.")
        return
        
    price_series = master_matrix[primary_ticker].dropna()
    if price_series.empty:
        st.warning("Insufficient data for recommendations.")
        return
        
    rolling_series = rolling_returns_matrix[primary_ticker] if rolling_returns_matrix is not None else None
    
    with st.spinner("Arbitrating Signals..."):
        engine = ArbitrationEngine(price_series, rolling_series)
        hero_state = engine.evaluate_current_state()
        
        current_price = price_series.iloc[-1]
        peak_price = price_series.max()
        action_grid = engine.generate_action_grid(hero_state['z_score'], current_price, peak_price)
        
        inversion_matrix = engine.run_inversion_matrix()

    # --- Section A: Arbitration Hero Panel ---
    st.markdown("### 🔹 Arbitration Hero Panel")
    
    state_color = "normal"
    if "ACCUMULATION" in hero_state['state']:
        state_color = "inverse" # Streamlit uses inverse for green/red alerts depending on theme, or we can use markdown
        st.success(f"### {hero_state['state']}")
    elif "DISTRIBUTION" in hero_state['state']:
        st.error(f"### {hero_state['state']}")
    else:
        st.info(f"### {hero_state['state']}")
        
    col1, col2, col3 = st.columns(3)
    col1.metric("Context: Z-Score", f"{hero_state['z_score']:.2f} σ")
    trend_str = "Positive" if hero_state['is_positive_trend'] else "Negative"
    col2.metric("Context: Trend (50>200 SMA)", trend_str)
    col3.metric("Veto Check", hero_state['veto_status'])

    with st.expander("ℹ️ How to read the Hero Panel", expanded=False):
        st.markdown("""
        - **Z-Score (Cheapness):** Measures how 'stretched' the current return is relative to history. 
            - Below **-1.5** means the asset is historically 'cheap' (Accumulation Zone).
            - Above **+1.5** means the asset is historically 'expensive' (Distribution Zone).
        - **Trend (Market Health):** Uses the 50-day and 200-day moving averages.
            - **Positive:** Short-term health is improving (Strong Market).
            - **Negative:** Short-term health is declining (Weak Market).
        - **Veto Check:** The system's final audit. It ensures we aren't buying a 'Falling Knife' or selling a 'Super-Trend' without confirmation.
        """)
    
    st.divider()
    
    # --- Section B: Action Grid ---
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

    with st.expander("ℹ️ Understanding the Action Grid", expanded=False):
        st.markdown("""
        The grid helps you manage **two different types of capital**:
        - **Monthly SIP Flow (Income):** This is your recurring investment. 
            - When an asset is cheap, the system suggests a **multiplier (e.g., 1.5x)** to capture the discount. 
            - When expensive, it suggests **scaling down** to avoid overpaying.
        - **Dry Powder (Lump Sum Reserves):** This is extra cash sitting in your bank.
            - It is **not** deployed during normal markets. 
            - It only triggers during deep 'crashes' (Drawdowns) to lower your average purchase price significantly.
        """)

    st.divider()
    
    # --- Section C: Inversion Transparency Matrix ---
    st.markdown("### 🔹 Inversion Transparency Matrix")
    st.info(f"**Veto Analysis:** {inversion_matrix['veto_printout']}")

    with st.expander("ℹ️ What is 'Inversion'?", expanded=False):
        st.markdown("""
        The 'Inversion' engine is a **reality check**. It looks back at every single time in the last 20+ years that the data looked *exactly* like it does today.
        
        - **Why?** To see if the signal was a 'False Positive' (a Value Trap).
        - **False Positive Rate:** If this is 20%, it means that in 1 out of 5 cases, the market continued to fall after this signal.
        - **Goal:** To give you the confidence to execute when the data is in your favor, and the humility to stay cautious when history warns of risk.
        """)
    
    with st.expander(f"View Historical False Positives ({inversion_matrix['events_found']} Events)", expanded=False):
        if inversion_matrix['events_found'] > 0:
            df_hist = inversion_matrix['history_table'].copy()
            # Format the output for readability
            if 'Next 3M Return' in df_hist.columns:
                df_hist['Next 3M Return'] = df_hist['Next 3M Return'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            if 'Next 6M Return' in df_hist.columns:
                df_hist['Next 6M Return'] = df_hist['Next 6M Return'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            if 'Max Drawdown (Next 1Y)' in df_hist.columns:
                df_hist['Max Drawdown (Next 1Y)'] = df_hist['Max Drawdown (Next 1Y)'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            if 'Z-Score' in df_hist.columns:
                df_hist['Z-Score'] = df_hist['Z-Score'].apply(lambda x: f"{x:.2f}")
                
            st.table(df_hist)
        else:
            st.info("No comparable historical events found for the current threshold.")
