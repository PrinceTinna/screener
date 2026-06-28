import streamlit as st


def render_help_guide():
    """Renders the full help and methodology guide in a dedicated tab."""

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
        <h2 style="margin:0; color:#fff;">📖 QuantPro Dashboard — User Guide</h2>
        <p style="margin:0.5rem 0 0; color:#aac8e8; font-size:0.95rem;">
            A practical reference for interpreting every metric, chart, and control in this dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: The Big Picture ────────────────────────────────
    with st.expander("🌐 What is QuantPro doing under the hood?", expanded=True):
        st.markdown("""
        QuantPro is a **Rolling Return Analysis Engine** for the Indian market (NSE/BSE). \
        Rather than looking at a single price point, it asks a sharper question:

        > *"If you had bought this asset **N days ago**, what would your return be today — \
        and how does that compare to every other point in history?"*

        **The core workflow:**
        1. **Data Lake** — Historical OHLC data for 21+ liquid Indian ETFs and Indices is fetched \
           from Yahoo Finance and cached locally as Parquet files from 2001 onward.
        2. **Matrix Engine** — All 21 assets are processed simultaneously as a single 2D NumPy array \
           (rows = trading days, columns = assets) using **VectorBT** for vectorised speed.
        3. **Inception Masking** — Newly launched ETFs (e.g., Silver BeES launched 2022) are \
           automatically masked with `NaN` before their inception date, preventing phantom return signals.
        4. **TR vs PR Routing** — ETFs use **Adjusted Close** (Total Return, capturing dividends). \
           Indices use **Close** (Price Return, no dividends). This is auto-detected from `universe.json`.
        5. **Cross-Sectional Percentiles** — At every point in time, the engine ranks the current \
           asset against the entire universe to build the percentile bands you see on the chart.
        """)

    # ── Section 2: Control Panel ──────────────────────────────────
    with st.expander("🎛️ Control Panel (Sidebar)"):
        st.markdown("""
        | Control | What it does |
        |---|---|
        | **Primary Asset Selector** | The main asset being analysed. All KPIs and charts are computed for this asset. |
        | **Benchmark Overlays** | Up to N assets rendered as faded lines on the time-series chart for direct visual comparison. |
        | **Rolling Window** | The look-back horizon for calculating returns. **1M** = 21 trading days, **1Y** = 252, **5Y** = 1,260. |

        ---
        #### Choosing the Right Rolling Window

        | Window | Best used for | Interpretation |
        |---|---|---|
        | **1M / 3M** | Short-term momentum, tactical trades | High noise; shows recency bias |
        | **6M** | Medium-term trend confirmation | Balances noise vs. signal |
        | **1Y** | Preferred baseline for ETF analysis | One full market cycle |
        | **3Y / 5Y** | Long-term wealth creation, SIP returns | Smooths out bear market dips |

        > 💡 **Pro Tip:** For sector rotation decisions, compare all sectors using the **1Y window** \
          to identify which sector is at a cyclically high or low rolling return.
        """)

    # ── Section 3: KPI Cards ──────────────────────────────────────
    with st.expander("📊 KPI Cards — Row 1"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
            #### Current Rolling Return
            <span class="badge">FORMULA</span> `(Price_today / Price_{today - N days}) - 1`

            The raw, un-annualized return over the selected window. This is **what you would have \
            earned** if you had bought N days ago and held until today.

            **How to interpret:**
            - A **positive value** means the asset is currently in profit for that window.
            - Compare it against the **Universe Median** on the chart to see if it's above or below average.
            - For a 1Y window: values > 15% indicate strong momentum; < 0% indicates a drawdown regime.
            """, unsafe_allow_html=True)

            st.markdown("""
            #### Annualized CAGR
            <span class="badge">FORMULA</span> `(1 + Rolling_Return)^(252 / window_days) - 1`

            Converts the raw rolling return into an **annualized growth rate**, making all windows \
            directly comparable regardless of their duration.

            **How to interpret:**
            - A 3M window return of 8% annualizes to ~36% CAGR — helpful for comparing short-term \
              bursts to long-term averages.
            - A 5Y CAGR > 12% for a broad index ETF is considered strong in the Indian market context.
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            #### Historical Win Rate
            <span class="badge">FORMULA</span> `Count(Rolling_Return > 0) / Total_Days`

            The **percentage of all historical rolling windows** that were profitable. This is a \
            probabilistic view of how often this asset delivered positive returns for the chosen window.

            **How to interpret:**
            - **> 70%:** Very consistent asset; low probability of loss for this window.
            - **50–70%:** Moderate consistency; returns depend on entry timing.
            - **< 50%:** More often negative than positive — suggests high volatility or structural decline.
            - Nifty 50 typically has a **~75% Win Rate on 1Y windows** historically.
            """, unsafe_allow_html=True)

            st.markdown("""
            #### Universe Rank
            <span class="badge">FORMULA</span> `Rank of today's return among all assets at today's date`

            Shows where the asset stands **right now** relative to the entire 21-asset universe. 
            
            **How to interpret:**
            - <span class="badge badge-green">#1–7</span> Top tercile — asset is outperforming most of the universe.
            - <span class="badge badge-orange">#8–14</span> Middle tercile — in line with the average.
            - <span class="badge badge-red">#15–21</span> Bottom tercile — underperforming; potential mean-reversion opportunity.
            """, unsafe_allow_html=True)

    # ── Section 4: Expectation Metrics ──────────────────────────
    with st.expander("🔮 Expectation Metrics — Row 2"):
        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("""
            #### Mean Rolling Return
            The mathematical average of every historical point-to-point return for this window.
            - **Use Case:** This is your primary "Expectation" if you pick a random entry date.
            
            #### P50 (Median) Return
            The middle value of history. Unlike Mean, this is not distorted by extreme crashes or rallies.
            - **Use Case:** A more "honest" baseline for what to expect in a typical year.
            """, unsafe_allow_html=True)
        with col_d:
            st.markdown("""
            #### P30 (Conservative) Return
            The level that the asset historically **exceeded 70% of the time**. 
            - **Use Case:** Helpful for "Stress Testing" or "Safe Withdrawal" planning. If your target is P30, you have high statistical confidence.
            
            #### Current vs Mean Delta
            The gap between today's performance and the long-term historical average.
            - <span class="badge badge-red">Positive Delta</span> The asset is currently "hot" (returning more than its average).
            - <span class="badge badge-green">Negative Delta</span> The asset is "cooled off" — potentially a better entry point from a valuation perspective.
            """, unsafe_allow_html=True)

    # ── Section 4: Rolling Returns Chart ─────────────────────────
    with st.expander("📈 Rolling Returns Chart — Row 2"):
        st.markdown("""
        The main chart is a **time-series of rolling returns** for every trading day in the asset's history.

        #### What each visual element means:

        | Element | Colour | Meaning |
        |---|---|---|
        | **Bold blue line** | `#007FFF` | The **Primary Asset's** rolling return at each historical date |
        | **Shaded grey band** | `rgba(128,128,128,0.2)` | The **25th–75th interquartile range** of the entire universe. This is the "normal zone" |
        | **Dashed grey line** | `rgba(128,128,128,0.5)` | The **Universe Median** — the 50th percentile across all 21 assets |
        | **Red faded lines** | `rgba(200,100,100,0.6)` | **Benchmark Overlays** selected in the sidebar |

        ---
        #### How to read the percentile bands
        The grey band is derived by computing the **cross-sectional percentiles** of the universe \
        at each point in time using a Numba-accelerated kernel.

        - When the **blue line is above the grey band**, the asset is in the **top 25%** of the universe — \
          historically a sign of momentum.
        - When the **blue line is inside the grey band**, the asset is in the **middle 50%** — neutral.
        - When the **blue line is below the grey band**, the asset is in the **bottom 25%** — potential \
          mean-reversion long entry if the asset has strong fundamentals.

        ---
        #### Reading the chart tactically
        1. **Identify the trend** — Is the blue line consistently above or below the median?
        2. **Check extremes** — Rolling returns far above the 75th band may indicate an asset \
           that is extended and due for a correction.
        3. **Use benchmarks** — Add the Nifty 50 (`^NSEI`) as a benchmark overlay to see \
           if a sector is outperforming or lagging the broad market.

        > ⚠️ **TR vs PR Warning:** If you compare an ETF (Total Return) with an Index (Price Return), \
          the ETF will appear to outperform over the long run simply because it includes reinvested dividends. \
          The dashboard will automatically warn you when this mismatch occurs.
        """)

    # ── Section 5: Distribution Chart ────────────────────────────
    with st.expander("📉 Return Distribution Chart — Row 3 (Left)"):
        st.markdown("""
        The **Return Distribution** is a probability density histogram of all historical rolling \
        returns for the selected asset and window.

        #### What to look for:

        | Pattern | What it means |
        |---|---|
        | **Tall, narrow bell curve** | Consistent, predictable returns (e.g., Liquid BeES) |
        | **Wide, flat distribution** | High volatility — large drawdowns are historically common |
        | **Right-skewed (tail on the right)** | More frequent large positive returns — positively skewed assets like Gold |
        | **Left-skewed (tail on the left)** | Crash risk — large negative returns occur more than a normal distribution would predict |
        | **Bimodal (two humps)** | The asset has two distinct return regimes (e.g., pre/post a structural shift) |

        #### The current rolling return marker
        The blue bar cluster containing **today's rolling return** tells you how exceptional \
        (or ordinary) the current return is vs. all of history. If today's return sits in the \
        far right tail, the asset is at a historical peak.

        > 💡 **Use Case:** If the distribution is left-skewed AND the current return is in the \
          right tail, consider this a **cautious signal** — mean reversion risk is elevated.
        """)

    # ── Section 6: Risk Analytics Table ──────────────────────────
    with st.expander("🔬 Risk Analytics Table — Row 3 (Right)"):
        st.markdown("""
        The Risk Analytics table is generated by **VectorBT's native stats engine** applied \
        to the rolling return time-series. Key metrics explained:

        | Metric | Formula / Definition | Interpretation |
        |---|---|---|
        | **Start / End** | Dataset date range | Data quality boundary |
        | **Period** | Total days analysed | Longer = more statistically significant |
        | **Total Return** | Compound growth of the series | Absolute wealth creation |
        | **Annualized Return** | CAGR of the full series | Comparable annualized baseline |
        | **Annualized Volatility** | Std Dev × √252 | Higher = riskier; Gold ~15%, Nifty 50 ~18% |
        | **Max Drawdown** | Worst peak-to-trough loss | Risk tolerance benchmark; keep < 50% for most |
        | **Sharpe Ratio** | (Return - Rf) / Volatility | > 1 is good; > 2 is excellent |
        | **Sortino Ratio** | Like Sharpe, but only penalises downside volatility | More relevant for asymmetric assets like Gold |
        | **Tail Ratio** | 95th percentile return / 5th percentile return | > 1 means upside tails are larger than downside |
        | **Kurtosis** | Measure of "fat tails" | > 3 means crashes are more probable than a normal distribution predicts |
        """)

    # ── Section 7: TR vs PR Explained ────────────────────────────
    with st.expander("⚖️ Total Return (TR) vs Price Return (PR) — Critical Concept"):
        st.markdown("""
        This is the **most commonly misunderstood concept** in Indian ETF analysis.

        | Type | Examples | Price Source | Includes Dividends? |
        |---|---|---|---|
        | **Total Return (TR)** | JUNIORBEES.NS, MID150BEES.NS, GOLDBEES.NS | `Adj Close` | ✅ Yes |
        | **Price Return (PR)** | ^NSEI, ^BSESN, ^CNXIT | `Close` | ❌ No |

        #### Why this matters:
        Over a 20-year horizon, dividend reinvestment can account for **30–40% of total wealth** for \
        a Nifty ETF. If you compare `JUNIORBEES.NS` (TR) with `^NSEI` (PR) over 10 years, the ETF \
        will appear to outperform the index significantly — but this is NOT alpha. It is simply dividends.

        #### Rule of thumb:
        - **Compare ETFs with ETFs** (TR vs TR) for performance evaluation.
        - **Compare Indices with Indices** (PR vs PR) for benchmark analysis.
        - Only compare TR vs PR if you want to visualize the **dividend contribution** explicitly.

        The dashboard will show a ⚠️ warning banner whenever you mix TR and PR assets.
        """)

    # ── Section 8: Asset Universe Guide ──────────────────────────
    with st.expander("🗺️ Asset Universe — Category Guide"):
        st.markdown("""
        | Category | Assets | Best used for |
        |---|---|---|
        | **Broad Market** | ^NSEI, ^BSESN, NIFTYBEES, JUNIORBEES, MID150BEES, HDFCSML250 | Macro regime analysis, core portfolio comparison |
        | **Sectoral** | ^NSEBANK, BANKBEES, ^CNXIT, ITBEES, ^CNXPHARMA, PHARMABEES, ^CNXFMCG, ^CNXAUTO, AUTOBEES | Sector rotation strategy, identifying leadership |
        | **Smart Beta** | MOM100 (Momentum 30) | Factor exposure; use 1Y rolling returns to time factor cycles |
        | **Commodities** | GOLDBEES, SETFGOLD, SILVERBEES | Safe-haven and inflation hedge analysis |
        | **Fixed Income / Cash** | LIQUIDBEES | Benchmark for the risk-free rate in India; returns ~6-7% p.a. |

        #### Typical analysis workflows:
        1. **Sector Rotation** — Set window to 1Y. Compare all sector indices. Move capital toward \
           sectors in the bottom 25th percentile with improving momentum.
        2. **Gold Hedge Timing** — Compare GOLDBEES vs ^NSEI. When NSEI is in its top 10% rolling \
           return and GOLDBEES is lagging, equity risk is elevated.
        3. **Liquid BeES Baseline** — Any asset with a Current Rolling Return below LIQUIDBEES for \
           the same window is delivering sub-risk-free returns. Avoid or exit.
        """)

    # ── Section 9: FAQ ────────────────────────────────────────────
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        **Q: Why does MOM100.NS (Momentum) only show data from 2022?**
        > A: The ETF was launched on February 10, 2022. All prior dates are correctly masked \
          as `NaN` by the inception guardrail in the pipeline. The dashboard will warn you if you \
          select a rolling window (e.g., 3Y) that predates the inception.

        **Q: Why is the 1Y Win Rate for LIQUIDBEES only 100%?**
        > A: Liquid BeES is a liquid fund ETF that tracks the overnight rate. Its price essentially \
          only goes up monotonically. It is not an investment vehicle for capital appreciation; \
          it is used as a cash parking and minimum return benchmark.

        **Q: What does it mean when the blue line is below the grey band for a long time?**
        > A: The asset is in the **bottom 25% of the universe** for the selected window. This can \
          signal a structural underperformer (avoid), or a **deep value mean-reversion opportunity** \
          (buy carefully with confirmation). Never trade on this signal alone.

        **Q: The Risk Analytics table shows an error or is blank. Why?**
        > A: VectorBT's stats engine requires a well-formed time-series with frequency metadata. \
          For newly launched ETFs with very short histories (< 252 days), the stats engine may \
          fail gracefully. Select a shorter rolling window (e.g., 1M or 3M) or choose a more \
          established asset.

        **Q: Why is the KPI showing a rank like "#3 of 8" instead of "#3 of 21"?**
        > A: The rank denominator is the number of assets with a **valid (non-NaN)** rolling return \
          today. For short-history ETFs like HDFCSML250 (launched 2023), the 3Y or 5Y rolling \
          return is `NaN` and they are excluded from the cross-sectional rank.
        """)


def render_sidebar_help():
    """Renders a concise help reference in the sidebar."""
    with st.sidebar.expander("❓ Quick Reference", expanded=False):
        st.markdown("""
        **Rolling Return:**
        Return earned if bought N days ago.
        
        **CAGR:**
        Annualized equivalent of the rolling return.
        
        **Win Rate:**
        % of historical windows with positive return.
        
        **Mean/Median:**
        Your mathematical "expected" return if entering randomly.
        
        **P30 (Conservative):**
        The return level exceeded 70% of the time.
        
        **Grey Band:**
        25th–75th percentile of the universe.
        
        **Median Line (dashed):**
        50th percentile of the universe.
        
        **TR ETF vs PR Index:**
        ETF includes dividends; Index does not.
        Use same type for fair comparison.
        
        ---
        *Switch to the **📖 Help & Guide** tab for full documentation.*
        """)
