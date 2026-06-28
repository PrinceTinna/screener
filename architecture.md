# Product Specification: QuantPro (Phase 1.8 - Global Multi-Asset & Fundamentals)
**Module:** Global ETF, Factor, Debt, & Index Rolling Return Analysis Station  
**Target Audience:** Quant Engineers, Systematic Traders, Macro Analysts  
**Tech Stack:** Python 3.9+, **vectorbt (0.28.2)**, **Numba**, Streamlit, **Plotly (6.7.0)**, yfinance, PyArrow

## 1. Product Vision & Scope
QuantPro Phase 1 delivers a global multi-asset rolling return dashboard tracking Indian, smart-beta, fixed-income sovereign debt, and international broad market assets. The backend architecture functions as an N-dimensional matrix processing engine using `vectorbt` and Numba. While the frontend visually profiles one primary asset at a time, the core engine processes the entire global multi-asset macro universe simultaneously via matrix broadcasting. 

In Phase 1.5+, the engine transcends simple descriptive statistics by integrating **Directionality (Slope)**, **Regime-Aware Logic**, **Bubble Detection**, and **Valuation Gates**. Signals are no longer static; they dynamically adapt to prevailing market volatility, valuations, and bubbles, ensuring that we buy the "bounce" rather than the "crash" (Falling Knife protection) and de-escalate allocations near historical valuation peaks (Overvaluation Veto).

---

## 2. The Segmented Global Asset Universe (`universe.json`)
The system strictly parses a pre-defined JSON registry to route API calls (Adj Close vs. Close) and validate inception dates. Assets are categorized by Segment (e.g. Broad Market, Sectoral, Thematic, International, Commodities, Fixed Income) to prevent invalid cross-sectional percentiles.

**Schema Example:**
```json
{
  "^NSEI": {
    "name": "NIFTY 50 Index",
    "class": "Broad Market - Benchmark",
    "type": "PR",
    "inception": "1996-04-22"
  },
  "JUNIORBEES.NS": {
    "name": "Nippon India ETF Nifty Next 50 Junior BeES",
    "class": "Broad Market - Equity",
    "type": "TR",
    "inception": "2003-02-21"
  },
  "LIQUIDBEES.NS": {
    "name": "Nippon India ETF Nifty 1D Rate Liquid BeES",
    "class": "Fixed Income - Cash",
    "type": "TR",
    "inception": "2003-07-08"
  }
}
```

---

## 3. Directory Layout (Phase 1.8 Updates)
```text
├── config/               
│   ├── settings.py       # Global constants (Trading days = 252)
│   └── universe.json     # The ETF/Index registry
│
├── data/                 
│   ├── fetcher.py        # yfinance logic (.NS/.BO suffix handling) + daily incremental fundamentals update
│   ├── pipeline.py       # Indian calendar alignment, Spike Filtering & master fundamentals matrix construction
│   ├── fundamentals_seed.py # [NEW] Historical baseline seed generator using price history and sector baselines
│   └── cache_manager.py  # Local parquet lifecycle management
│
├── core/                 
│   ├── indicators.py     # Custom vbt.IndicatorFactory (Rolling Return, CAGR)
│   ├── features.py       # Numba-compiled math kernels (Percentiles)
│   ├── vbt_engine.py     # VectorBT orchestration layer
│   ├── validators.py     # Matrix shape and inception date checks
│   └── state_math.py     # OLS Slope (β), Z-Score, Trend, 2D Regime Classification, Bubble Z-Score
│
├── strategies/           # Multi-asset signal generation layer
│   ├── rolling_returns.py
│   └── arbitration.py    # Logic for multipliers, drawdowns, and Inversion Veto
│
├── ui/                   
│   ├── app.py            # Master tab routing, global state, and fundamentals caching
│   ├── components.py     
│   └── views/            # Modular dashboard views
│       ├── screener.py         # [Tab 0] Global Discovery Grid with P/E and YoY EPS Growth columns
│       ├── asset_dashboard.py  # [Tab 1] Pure historical plotting + Regime Shading + Dual-Y Valuation Charts
│       ├── signals.py          # [Tab 2] Regime-Aware Hysteresis Signal Engine with Valuation Gate & Hover Tooltips
│       └── help_guide.py       # [Tab 3] Documentation
│
├── tests/                
│   ├── test_vbt_math.py  
│   ├── test_pipeline.py  # [UPDATED] Matrix shape, Spike filter, and fundamentals alignment checks
│   ├── test_arbitration.py 
│   ├── test_signals.py   # OLS Slope, Hysteresis, Hard Veto, Bubble Z-score, and Valuation Gate tests
│   ├── test_ui_tabs.py   
│   └── test_ui.py        
```

---

## 4. Data Pipeline & Indian Market Guardrails
Indian market data introduces unique anomalies. The `data/pipeline.py` must enforce the following before creating the master `vectorbt` matrix:

1.  **Muhurat Trading Anomaly:** [KNOW CONSTRAINT] The pipeline should identify the 1-hour special Diwali trading sessions. Currently handled via duplicate removal and ffill.
2.  **Dynamic Holiday Calendar:** Uses the `holidays.India()` library to filter out non-trading days (Holi, Diwali, Eid) before building the master matrix.
3.  **Total Return (TR) vs. Price Return (PR) Routing:** `fetcher.py` checks `universe.json`. If type is `TR` (ETFs), fetch `Adj Close`. If type is `PR` (Indices), fetch `Close`.
4.  **Spike Filtering Guardrail:** Statistical guardrail using a 21-day rolling median to mask and forward-fill price anomalies (>30% move from median).

---

## 5. Core Mathematical Engine (Numba / VectorBT)
All formulas are computed simultaneously across the 2D matrix.

**1. Rolling Metrics & Absolute Context:**
* **Rolling Return:** $R_{t, w} = \frac{P_t}{P_{t-w}} - 1$
* **Excess Return:** $Excess_{t, w} = R_{t, w} - LiquidBees_{t, w}$
* **Z-Score:** $Z_{t, w} = \frac{R_{t, w} - \mu(R_{w})}{\sigma(R_{w})}$
* **[NEW] Return Slope (Linear Regression β):** $\beta_{t} = \text{Linear Regression Slope of } R_{t, w} \text{ over } n \text{ days}$. This replaces the primitive $\Delta R$ to provide a stable, noise-filtered measure of momentum directionality and trend strength.

**2. Long-Term Bubble Z-Score ($Z_{bubble}$):** Measures structural price divergence from its long-term baseline.
* **Price Distance from Trendline ($D_t$):** $D_t = \frac{P_t - SMA_{t, 756}}{SMA_{t, 756}}$ (using 3-year/756-day SMA).
* **Bubble Z-Score ($Z_{bubble, t}$):** $Z_{bubble, t} = \frac{D_t - \mu_{lifetime}(D)}{\sigma_{lifetime}(D)}$ where the mean ($\mu_{lifetime}$) and standard deviation ($\sigma_{lifetime}$) are calculated over the entire available lifetime history of the asset.
* **Classification:**
    * *Normal*: $Z_{bubble} < 1.5$ (🟢)
    * *Extended*: $1.5 \le Z_{bubble} < 2.0$ (🟡)
    * *2-Sigma Bubble*: $2.0 \le Z_{bubble} < 3.0$ (🟠)
    * *3-Sigma Superbubble*: $Z_{bubble} \ge 3.0$ (🔴)

**3. Fundamentals & Valuation Metrics [NEW]:**
* **Daily Derived EPS:** $EPS_t = \frac{P_t}{PE_t}$ (where daily P/E is loaded from fundamentals cache).
* **Year-over-Year (YoY) EPS Growth:** $EPS\_Growth_t = \frac{EPS_t}{EPS_{t-252}} - 1$.
* **Benchmark Asset Mapping:** ETFs (like `NIFTYBEES.NS`) share benchmark index fundamentals (like `^NSEI`) to guarantee completeness. Commodities and cash equivalents are assigned explicit `NaN` values.
* **Daily Extrapolated Fallback:** If `yfinance` live fundamentals are offline, updates extrapolate using a smooth 12% default annual EPS growth rate from the last valid baseline cache.

**4. Regime Classification Engine:**
Market regimes are defined by a 2D matrix of rolling volatility ($\sigma_{t, w}$) and price trend.
* **Volatility Dimensions:** Low (<40th), Neutral (40th-75th), High (>75th), and **Extreme (>90th)**.
* **Trend Dimensions:** Up (Current $R_{t, w} > SMA$), Down (Current $R_{t, w} < SMA$).
* **Regime Mapping:**
    * **Trending Bull:** Low Vol + Up Trend.
    * **Panic / Liquidation:** High Vol + Down Trend.
    * **Recovery:** High Vol + Up Trend.
    * **Mean-Reverting Range:** Low Vol + Flat Trend.

**5. Performance & Compilation Caching Guardrails:**
To prevent latency bottlenecks during app-wide reruns when the user changes sidebar options, the core engine enforces:
* **JIT Caching (`cache=True`):** All Numba compilation kernels (`fast_percentiles_2d` and `_ols_slope_numba`) are cached to disk to eliminate the 2-second startup compilation lag.
* **Vectorized Tail Math:** Cross-sectional screener metrics (volatility, Z-score) bypass the costly historical calculation loops by computing tail statistics directly on `returns_matrix.tail(window_days)`.
* **Batched Plotly Layout Shading:** Background volatility regime rendering is optimized by batching shapes and annotations in-memory and committing them in a single `fig.update_layout` call, bypassing Plotly's sequential layout-validation overhead.

---

## 6. Dashboard UX/UI Specification

### A. The Screener (Cross-Sectional Discovery Matrix)
**Objective:** A top-down global grid to spot risk-adjusted opportunities.
* **Global Filters:** Segment Toggle, Signal Filter.
* **[UPDATED] Data Grid Columns:**
    * Asset Name & Segment
    * 3Y Current CAGR
    * 3Y Volatility ($\sigma$)
    * 3Y Max Drawdown
    * 3Y Sharpe Ratio (Using Dynamic $R_f$)
    * 3Y Z-Score & 3Y Segment Rank
    * Bubble Risk Badge (🟢 Normal / 🟡 Extended / 🟠 2-Sigma Bubble / 🔴 3-Sigma Superbubble)
    * **[NEW] P/E Ratio** (Sourced from fundamental caches; NaNs for Commodities/Cash)
    * **[NEW] YoY EPS Growth (%)** (Trailing 12-month corporate earnings momentum)
    * Regime-Adjusted Signal Badge (🟢/🟡/🔴)
* **Drill-Down Routing:** Single-click sets URL parameters and routes to the Asset Dashboard.

### B. Global Control Panel (Sidebar)
* **Asset Class Filter**, **Primary Asset Selector**, **Benchmark Overlays**, **Rolling Window**, and **Historical Lookback**.

### C. Tab 1: Asset Dashboard (Pure Analysis)
* Strict historical profiling utilizing **Volatility/Bubble Regime Shading** on the main rolling return chart.
* **[NEW] Valuation & Fundamentals Sub-Tabs:** Provides a dual-view selector:
    *   **Tab 1 (P/E & YoY EPS Growth):** Dual-y-axis Plotly chart displaying daily P/E Ratio (left y-axis) alongside YoY EPS Growth (right y-axis).
    *   **Tab 2 (Price vs. EPS Rolling Returns):** Single-axis return decomposition displaying Total Price Return, Fundamental Return (EPS Growth), and Speculative Return (Multiple Expansion). It draws 10th, 50th, and 90th percentile historical bounds for Speculative Return to identify extreme sentiment extensions.

### D. Tab 2: Signals (Execution Layer)
**Objective:** Operationalize analytical data into rules-based directives, explicitly conditioned by current market volatility regime and trend direction.

* **Section A: The Regime-Aware Signal Logic Engine**
    * **Current Regime Display:** Explicitly states the 2D environment (e.g., *"Current Regime: Recovery (High Vol + Up Trend). Prioritizing Volatility-Adjusted Momentum."*)
    * **Hard Veto Condition:** If Volatility > 90th percentile of history, all **Momentum** signals are disabled entirely (Market Breakdown Regime).
    * **Bubble De-escalation Rule:** If $Z_{bubble} \ge 2.0$, any active momentum signal is downgraded to `🟡 Weak Momentum (Bubble Risk)` to prevent over-allocation at extreme peaks.
    * **[NEW] Valuation Gate Veto:** If the current P/E ratio is in the top 10% of historical values (percentile $\ge 90\%$), any active momentum signal is downgraded to `🟡 Weak Momentum (Overvaluation Risk)` accompanied by a high-priority warning banner.
    * **[NEW] Mathematical Hover Explanations:** The metric elements include detailed parameter tooltips, and a layout expander provides complete formulas and conditions explaining exactly how the state machine derived today's execution signal.
    * 🟢 **Momentum State (Prioritizing Hysteresis for Stability):**
        * **Entry:** Segment Rank > 75th Percentile AND Z-Score > 0.5 AND Slope ($\beta$) $\ge$ 0.
        * **Exit (Hysteresis):** Segment Rank < 60th Percentile OR Z-Score < 0.
        * *Regime Modifier:* If triggered during a High Vol Regime, downgrade badge to 🟡 *Weak Momentum (Whipsaw Risk)*.
    * 🔴 **Reversal Zone (Prioritized in High Vol):**
        * Segment Rank < 25th Percentile, Z-Score < -1.0.
        * Slope ($\beta$) > 0 (Ensures the "falling knife" has stopped dropping and the first derivative has turned positive).
        * *Regime Modifier:* If triggered during a High Vol Regime, upgrade badge to 🟢 *Strong Reversal*.

* **Section B: The Action Grid (Flow vs. Reserves)**
    * SIP Flow Multipliers and Dry Powder Deployment triggers.

---

## 7. Automated Testing Specification (`pytest`)
* **Scenario 1–6:** Existing matrix broadcasting, Numba fallback, matrix alignment, holiday integrity, and UI state isolation tests.
* **Scenario 7: Falling Knife Guardrail (Unit Test) [NEW]:** Pass a synthetic array where an asset's price plummets to a Z-score of -2.0 but continues to drop (Slope < 0). Assert that the Reversal Signal does **not** trigger until the slope turns positive.
* **Scenario 8: Regime-Conditioned Downgrades (Unit Test):** Pass a synthetic array with extreme rolling returns (Rank 90th percentile) but massive expanding volatility (Vol > 80th percentile). Assert that the Momentum Signal is correctly downgraded due to the High Volatility regime.
* **Scenario 9: Hard Veto Execution (Unit Test) [NEW]:** Pass a synthetic price array with volatility exceeding the 90th percentile. Assert that all Momentum signals are disabled entirely, regardless of trend or rank strength.
* **Scenario 10: Bubble Risk De-escalation (Unit Test) [NEW]:** Pass a synthetic parabolic price series where $Z_{bubble} \ge 2.0$. Assert that any active momentum signal is correctly downgraded to `🟡 Weak Momentum (Bubble Risk)`.

---

## 8. Data Quality & Sanity Validation Specification
Detailed specifications and guidelines on data freshness, split-adjustment verification gates, and signal sanity safeguards are documented in [Data Quality Specification](file:///Users/princetinna/Documents/quant_dashboard2/data_quality_spec.md).

---

## 9. Changelog
* v1.3.0: Initial phase 1.3 architecture definition.
* v1.3.1: Complete instantiation of the Phase 1.3 Dashboard.
* v1.3.2: Expanded asset universe to 21+ liquid Indian Indices and ETFs.
* v1.3.3: Added Spike Filtering specification.
* v1.4.0: Major refactor splitting UI into MVC-style Tab routing. Added Execution Layer.
* v1.5.0: Integrated 2D Market Regimes (Volatility + Trend) and Signal Directionality via Linear Regression Slope ($\beta$).
* v1.6.0: Transitioned to production-grade signal engine. Added OLS Regression Slope for trend direction, state-machine Hysteresis for signal stability, and a Cross-Sectional Screener with drill-down URL routing. Added Scenario 9 (Hard Veto) to testing suite.
* v1.6.1: Implemented performance optimizations (JIT caching, batched Plotly layouts, and vectorized screener tail math) to eliminate rolling window and ticker switch latency.
* v1.7.0: Integrated 2-Sigma and 3-Sigma Bubble Detection framework. Added $Z_{bubble}$ detrended lifetime indicator, Screener column, Asset Dashboard KPI card/chart shading, and execution de-escalation rule.
* v1.8.0: [NEW] Sourced and cached historical P/E and YoY EPS Growth fundamentals. Integrated aligned fundamentals matrices into DataPipeline, added Screener data columns, added dual-axis P/E vs. EPS Growth charts to Asset Dashboard, implemented a Valuation Gate (percentile >= 90%) downgrade mechanism, and included interactive mathematical signal explanations in the UI.
* v1.8.1: Cleaned up all redundant tracking ETFs (Nifty BeES, Bank BeES, QQQ, SPY, CPSE ETF, IT BeES, etc.) to track core benchmark indices directly, avoiding pricing drift warnings and local exchange premiums. Purged dead legacy files (`core/universe.py`, `ui/views/rolling_analysis.py`, `ui/views/single_asset.py`, and `ui/views/recommendations.py`) to streamline the workspace. Refactored test assertions and benchmark-linked fundamentals mapping.


