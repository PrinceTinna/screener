import streamlit as st


def render_help_guide():
    """Renders the full help and methodology guide in a dedicated tab using a tabbed premium UI."""

    st.markdown("""
    <style>
    .help-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .badge {
        display: inline-block;
        background: rgba(0,127,255,0.25);
        color: #6db3ff;
        border: 1px solid rgba(0,127,255,0.4);
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 8px;
    }
    .badge-green { background: rgba(0,200,100,0.2); color: #5dffa5; border-color: rgba(0,200,100,0.4); }
    .badge-orange { background: rgba(255,165,0,0.2); color: #ffd77a; border-color: rgba(255,165,0,0.4); }
    .badge-red { background: rgba(220,50,50,0.2); color: #ff8a8a; border-color: rgba(220,50,50,0.4); }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Header ──────────────────────────────────────────────
    st.markdown("""
    <div class="help-header">
        <h2 style="margin:0; color:#fff;">📖 QuantPro — Methodology & User Guide</h2>
        <p style="margin:0.5rem 0 0; color:#aac8e8; font-size:0.95rem;">
            A complete reference guide for interpreting every metric, validation safeguard, and signal in the dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit Tabs for easy navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏁 Quick Start Playbook",
        "🚨 Signals & Quality Glossary",
        "📊 Metric Cheat Sheet",
        "⚖️ TR vs PR Concept",
        "🗺️ Asset Universe Guide",
        "❓ FAQs"
    ])

    with tab1:
        st.markdown("### 🏁 Tactical Investor Playbooks")
        st.markdown("Use these two core workflows to navigate regimes and optimize asset allocations:")

        st.info("""
        **Playbook A: The Sector Rotation Strategy (For ETF Investors)**
        1. Set the **Rolling Window** to **1Y** (preferred baseline for mid-term trend strength).
        2. View the **Global Discovery Matrix** (Screener tab) and filter by **Equity - Sector**.
        3. Identify sector ETFs in the **bottom 25th percentile** (`Rank (%)` < 25%) with a `Reversal` signal.
        4. Cross-verify that the sector's **P/E Ratio** is below its historical median.
        5. Allocate capital toward these mean-reverting sectors while trimming sectors displaying **Bubble Risk** in the top 10% returns.
        """)

        st.warning("""
        **Playbook B: Margin of Safety Filter (Tactical Momentum Entry)**
        1. Select your target equity asset (e.g. `JUNIORBEES.NS`).
        2. Verify the overall **Signal** is `🟢 Momentum`.
        3. Check **Expectation Metrics**: Ensure the **Current vs Mean Delta** is not excessively stretched (e.g., > 20%).
        4. Check the **Valuation Percentile**: If the current P/E is below the 80th percentile, you have a safe margin of safety.
        5. If the **Valuation Gate** triggers a warning (`Overvaluation Alert`), de-escalate standard allocation size to protect capital.
        """)

        st.markdown("""
        #### Under the Hood Engine Workflow
        1. **Data Lake:** Raw price OHLCV data is downloaded via Yahoo Finance API and saved locally in `data/cache` parquets.
        2. **Matrix Engine:** All assets are aligned and loaded into a single 2D NumPy array for vectorized speed via **VectorBT**.
        3. **Inception Masking:** Assets with short histories are automatically masked with `NaN` before their inception date to prevent false historical comparison signals.
        4. **TR vs PR Detection:** Capture dividends (`Adj Close`) for ETFs vs. pure price indices (`Close`) dynamically to avoid dividend comparison skew.
        """)

    with tab2:
        st.markdown("### 🚨 Signals, Guards & Valuation Glossary")
        st.markdown("Detailed breakdown of the signal states and the data validation guardrails:")

        st.markdown("#### Core Execution Signals")
        st.markdown("""
        | Signal State | Badge Color | Meaning & Action |
        |---|---|---|
        | **🟢 Momentum** | Success | Strong positive momentum (Z-Score > 1.5, OLS Slope > 0). Clear to accumulate. |
        | **🔴 Reversal** | Danger | Oversold bounce detected (Z-Score < -1.5, OLS Slope > 0). Potential value entry. |
        | **🟡 Neutral** | Warning | Normal operating regime. Stand by / Hold existing positions. |
        """)

        st.markdown("#### Data Quality & Valuation Guards")
        st.markdown("""
        *   **Check A: Minimum History Guard (`🟡 Insufficient History`)**
            *   *Rule:* Any asset with less than **252 trading days (~1 year)** of history cannot generate mathematical Z-scores or volatility metrics safely.
            *   *Behavior:* Locks the signal state to `Insufficient History` and disables buy/sell triggers.
        *   **Check B: Bubble Sample-Size Guard (`🟡 Insufficient History` / `Bubble Risk: NaN`)**
            *   *Rule:* Calculating lifetime trend standard deviation requires at least **1,000 trading days (~4 years)** of data.
            *   *Behavior:* If history < 1000 days, masks the $Z_{bubble}$ metric to `NaN`, overrides the bubble badge to `Insufficient History`, and bypasses bubble alert vetoes.
        *   **Check C: Fat-Tail Outlier CAGR Cap (`🔴 High Uncertainty (Data Gap)`)**
            *   *Rule:* If the rolling 1-year CAGR exceeds **150%** and the rolling window contains more than **5% missing data bars**, it signals a data gap.
            *   *Behavior:* Immediately downgrades the signal to `High Uncertainty (Data Gap)` to protect the investor from trading on dirty data feeds.
        *   **Check D: Cross-Sectional Sync Guard (`Exclude from Ranks`)**
            *   *Rule:* Compares the active asset's local cache timestamp against the master index. If it is lagging by **> 3 business days**, it is marked as stale.
            *   *Behavior:* Automatically sets its CAGR return to `NaN`, excluding it from the segment rank calculation to prevent stale rankings.
        *   **Check E: Live P/E Outlier Threshold (Simulated Index Fallback)**
            *   *Rule:* Compares the live yfinance P/E against the 3-year historical average.
            *   *Behavior:* If the live P/E deviates by **$> 5\sigma$**, it is rejected as a data feed error, and the engine falls back to simulated index valuations.
        *   **Valuation Gate (`🔴 Momentum - Overvaluation Alert`)**
            *   *Rule:* Triggers if the current asset P/E is in the **$\ge$ 90th historical percentile**.
            *   *Behavior:* De-escalates the momentum signal to warning state, prompting the user to trim/halt rather than buy the top.
        """)

    with tab3:
        st.markdown("### 📊 Metrics Reference Sheet")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h5>Current Rolling Return</h5>
                <span class="badge">FORMULA</span> <code>(Price_today / Price_{today - N days}) - 1</code>
                <p style="margin-top:0.5rem; font-size:0.875rem;">
                    The raw return earned over the selected lookback window. Represents your actual profit/loss if you held over this period.
                </p>
            </div>
            <div class="metric-card">
                <h5>Annualized CAGR</h5>
                <span class="badge">FORMULA</span> <code>(1 + Rolling_Return)^(252 / window_days) - 1</code>
                <p style="margin-top:0.5rem; font-size:0.875rem;">
                    Converts rolling returns into annualized growth rates. Essential for comparing return strength across different windows (e.g. comparing 3M to 5Y).
                </p>
            </div>
            <div class="metric-card">
                <h5>Historical Win Rate</h5>
                <span class="badge">FORMULA</span> <code>Count(Rolling_Return > 0) / Total_Days</code>
                <p style="margin-top:0.5rem; font-size:0.875rem;">
                    The percentage of all historical rolling windows that closed with positive returns. Measures asset consistency (Nifty 50 1Y Win Rate is ~75%).
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h5>Mean & P50 (Median) Return</h5>
                <span class="badge">EXPECTATION</span>
                <p style="margin-top:0.5rem; font-size:0.875rem;">
                    The mathematical average (Mean) and the midpoint (P50) of all historical returns. Median is robust to extreme tail events (market crashes or bubbles).
                </p>
            </div>
            <div class="metric-card">
                <h5>P30 (Conservative) Return</h5>
                <span class="badge">STRESS TEST</span>
                <p style="margin-top:0.5rem; font-size:0.875rem;">
                    The return level that the asset historically exceeded 70% of the time. Used for conservative scenario planning and withdrawal safety buffers.
                </p>
            </div>
            <div class="metric-card">
                <h5>Sharpe & Sortino Ratios</h5>
                <span class="badge">RISK ENGINE</span>
                <p style="margin-top:0.5rem; font-size:0.875rem;">
                    Sharpe measures excess return per unit of total risk. Sortino only penalizes negative volatility, making it superior for asymmetric assets like Gold.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### ⚖️ Total Return (TR) vs. Price Return (PR)")
        st.markdown("This is the most critical concept to understand when comparing ETFs with indexes:")

        st.warning("""
        **Total Return (TR) vs. Price Return (PR) Mismatch:**
        *   **Total Return (TR):** Captures both price change and reinvested dividends. Uses **Adjusted Close**. (e.g., `JUNIORBEES.NS`, `MID150BEES.NS`, `GOLDBEES.NS`).
        *   **Price Return (PR):** Tracks index price changes only, ignoring dividends. Uses standard **Close**. (e.g., `^NSEI`, `^BSESN`, `^CNXIT`).
        
        *Over a 10-year horizon, dividends can contribute up to 30-40% of total wealth. Comparing a TR ETF against a PR Index creates a false alpha bias. The dashboard will display a warning banner if you mix asset types.*
        """)

    with tab5:
        st.markdown("### 🗺️ Mapped Asset Universe Reference")
        st.markdown("The dashboard tracks 21 core index benchmarks and trackers. Mapped by asset class:")

        st.markdown("""
        | Asset Class | Ticker Symbol | Name | Description |
        |---|---|---|---|
        | **Equity - Broad** | `^NSEI` | Nifty 50 Index | Benchmark broad index (Price Return) |
        | **Equity - Broad** | `^BSESN` | S&P BSE Sensex Index | Benchmark broad index (Price Return) |
        | **Equity - Broad** | `JUNIORBEES.NS` | Nippon India Nifty Next 50 ETF | Tracks Nifty Next 50 (Total Return) |
        | **Equity - Broad** | `MID150BEES.NS` | Nippon India Nifty Midcap 150 ETF | Tracks Nifty Midcap 150 (Total Return) |
        | **Equity - Broad** | `HDFCSML250.NS` | HDFC Nifty Smallcap 250 ETF | Tracks Nifty Smallcap 250 (Total Return) |
        | **Equity - Sector** | `^NSEBANK` | Nifty Bank Index | Banking sector index (Price Return) |
        | **Equity - Sector** | `^CNXIT` | Nifty IT Index | Information Technology index (Price Return) |
        | **Equity - Sector** | `^CNXPHARMA` | Nifty Pharma Index | Pharmaceuticals index (Price Return) |
        | **Equity - Sector** | `^CNXFMCG` | Nifty FMCG Index | Consumer Goods index (Price Return) |
        | **Equity - Sector** | `^CNXAUTO` | Nifty Auto Index | Automobile sector index (Price Return) |
        | **Equity - Sector** | `^CNXPSE` | Nifty PSE Index | Public Sector Enterprises index (Price Return) |
        | **Equity - Sector** | `PSUBNKBEES.NS` | Nippon India Nifty PSU Bank ETF | Tracks PSU Bank Index (Total Return) |
        | **Equity - Sector** | `INFRABEES.NS` | Nippon India Nifty Infrastructure ETF | Tracks Infrastructure Index (Total Return) |
        | **Smart Beta** | `MOM100.NS` | Nippon India Nifty 50 Value 20 ETF | Smart beta factor tracker (Total Return) |
        | **Smart Beta** | `LOWVOL.NS` | ICICI Prudential Nifty Low Vol 30 ETF | Low-volatility factor tracker (Total Return) |
        | **Commodities** | `GOLDBEES.NS` | Nippon India Gold ETF | Tracks Gold bullion prices (Total Return) |
        | **Commodities** | `SILVERBEES.NS` | Nippon India Silver ETF | Tracks Silver bullion prices (Total Return) |
        | **Fixed Income** | `LICNETFGSC.NS` | LIC MF Nifty 8-13 yr G-Sec ETF | Long-term Sovereign Debt tracker (Total Return) |
        | **Cash Equivalent** | `LIQUIDBEES.NS` | Nippon India Liquid ETF | Tracks short-term money market rates (Total Return) |
        | **International** | `^NDX` | Nasdaq 100 Index | US Tech Index (Price Return) |
        | **International** | `EEM` | iShares MSCI Emerging Markets ETF | Emerging Markets Tracker (Total Return) |
        """)

    with tab6:
        st.markdown("### ❓ Frequently Asked Questions (FAQ)")

        st.markdown("""
        **Q: Why does MOM100.NS only show data from 2022?**
        > A: The ETF was launched on Feb 10, 2022. The Inception Guardrail automatically masks all historical data before this date with `NaN` to prevent calculation errors.
        
        **Q: Why is LIQUIDBEES's win rate consistently 100%?**
        > A: Liquid BeES tracks the overnight call money rate. Its NAV is designed to only compound positively, acting as a cash baseline.
        
        **Q: What should I do if the P/E Outlier check is active?**
        > A: The system will automatically use the simulated index reference calculations as a fallback to ensure your signals remain valid and uncorrupted.
        
        **Q: Why did my signal show "High Uncertainty"?**
        > A: This happens when the underlying pricing history has a gap (> 5% missing data) and a CAGR spike > 150%. The engine locks the signal to prevent false execution.
        """)


def render_sidebar_help():
    """Renders a concise help reference in the sidebar."""
    with st.sidebar.expander("❓ Quick Reference Guide", expanded=False):
        st.markdown("""
        **Rolling Return:**
        Return earned if bought N days ago.
        
        **CAGR:**
        Annualized equivalent of the rolling return.
        
        **Win Rate:**
        % of historical windows with positive return.
        
        **Grey Band:**
        25th–75th percentile of the universe.
        
        **Median Line (dashed):**
        50th percentile of the universe.
        
        **Signals Glossary:**
        *   `🟢 Momentum`: Strong bull trend.
        *   `🔴 Reversal`: Oversold bounce.
        *   `🟡 Neutral`: Stand by / Hold.
        *   `🟡 Insufficient History`: Short-history asset (< 252 days).
        *   `🔴 High Uncertainty`: Data gap detected.
        
        ---
        *Switch to the **📖 Help & Guide** tab for full documentation.*
        """)
