# Product Specification: QuantPro (Phase 1.6 - Global Screener & Hysteresis)
**Module:** Indian ETF & Index Rolling Return Analysis Station  
**Target Audience:** Quant Engineers, Systematic Traders, Macro Analysts  
**Tech Stack:** Python 3.9+, **vectorbt (0.28.2)**, **Numba**, Streamlit, **Plotly (6.7.0)**, yfinance, PyArrow

## 1. Product Vision & Scope
QuantPro Phase 1 delivers a strictly ETF/Index-focused rolling return dashboard tailored for the Indian markets (NSE/BSE). The backend architecture functions as an N-dimensional matrix processing engine using `vectorbt` and Numba. While the frontend visually profiles one primary asset at a time, the core engine processes the entire Indian macro universe simultaneously via matrix broadcasting. 

In Phase 1.5, the engine transcends simple descriptive statistics by integrating **Directionality (Slope)** and **Regime-Aware Logic**. Signals are no longer static; they dynamically adapt to prevailing market volatility, ensuring that we buy the "bounce" rather than the "crash" (Falling Knife protection).

---

## 2. The Segmented Indian Asset Universe (`universe.json`)
The system strictly parses a pre-defined JSON registry to route API calls (Adj Close vs. Close) and validate inception dates. Assets must be strictly categorized by Segment [Equity, Commodity, Fixed Income] to prevent invalid cross-sectional percentiles.

**Schema Example:**
```json
{
  "NIFTYBEES.NS": {
    "name": "Nippon India ETF Nifty 50 BeES",
    "class": "Equity - Large Cap",
    "type": "TR", 
    "inception": "2001-12-28"
  },
  "^NSEI": {
    "name": "NIFTY 50 Index",
    "class": "Equity - Benchmark",
    "type": "PR",
    "inception": "1996-04-22"
  },
  "BANKBEES.NS": {
    "name": "Nippon India ETF Nifty Bank BeES",
    "class": "Equity - Sectoral",
    "type": "TR",
    "inception": "2004-05-27"
  },
  "LIQUIDBEES.NS": {
    "name": "Nippon India ETF Liquid BeES",
    "class": "Cash Equivalent",
    "type": "TR",
    "inception": "2003-07-08"
  }
}
```

---

## 3. Directory Layout (Phase 1.6 Updates)
```text
├── config/               
│   ├── settings.py       # Global constants (Trading days = 252)
│   └── universe.json     # The strict Indian ETF/Index registry
│
├── data/                 
│   ├── fetcher.py        # yfinance logic (.NS/.BO suffix handling)
│   ├── pipeline.py       # Indian calendar alignment, Spike Filtering & matrix construction
│   └── cache_manager.py  # Local parquet lifecycle management
│
├── core/                 
│   ├── indicators.py     # Custom vbt.IndicatorFactory (Rolling Return, CAGR)
│   ├── features.py       # Numba-compiled math kernels (Percentiles)
│   ├── vbt_engine.py     # VectorBT orchestration layer
│   ├── validators.py     # Matrix shape and inception date checks
│   └── state_math.py     # [UPDATED] OLS Slope (β), Z-Score, Trend, 2D Regime Classification
│
├── strategies/           # Multi-asset signal generation layer
│   ├── rolling_returns.py
│   └── arbitration.py    # Logic for multipliers, drawdowns, and Inversion Veto
│
├── ui/                   
│   ├── app.py            # Master tab routing & global state
│   ├── components.py     
│   └── views/            # Modular dashboard views
│       ├── screener.py         # [Tab 0] NEW: Global Discovery Grid
│       ├── asset_dashboard.py  # [Tab 1] Pure historical plotting + Regime Shading
│       ├── signals.py          # [Tab 2] NEW: Regime-Aware Hysteresis Signal Engine
│       ├── recommendations.py  # [Legacy] Retained for backward compatibility
│       └── help_guide.py       # [Tab 3] Documentation
│
├── tests/                
│   ├── test_vbt_math.py  
│   ├── test_pipeline.py  
│   ├── test_arbitration.py 
│   ├── test_signals.py   # [NEW] OLS Slope, Hysteresis, Hard Veto tests
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

**2. [NEW] Regime Classification Engine:**
Market regimes are defined by a 2D matrix of rolling volatility ($\sigma_{t, w}$) and price trend.
* **Volatility Dimensions:** Low (<40th), Neutral (40th-75th), High (>75th), and **Extreme (>90th)**.
* **Trend Dimensions:** Up (Current $R_{t, w} > SMA$), Down (Current $R_{t, w} < SMA$).
* **Regime Mapping:**
    * **Trending Bull:** Low Vol + Up Trend.
    * **Panic / Liquidation:** High Vol + Down Trend.
    * **Recovery:** High Vol + Up Trend.
    * **Mean-Reverting Range:** Low Vol + Flat Trend.

---

## 6. Dashboard UX/UI Specification

### A. The Screener (Cross-Sectional Discovery Matrix)
**Objective:** A top-down global grid to spot risk-adjusted opportunities.
* **Global Filters:** Segment Toggle, Signal Filter.
* **[UPDATED] Data Grid Columns:**
    * Asset Name & Segment
    * 3Y Current CAGR
    * **[NEW] 3Y Volatility ($\sigma$)**
    * **[NEW] 3Y Max Drawdown**
    * **[NEW] 3Y Sharpe Ratio** (Using Dynamic $R_f$)
    * 3Y Z-Score & 3Y Segment Rank
    * Regime-Adjusted Signal Badge (🟢/🟡/🔴)
* **Drill-Down Routing:** Single-click sets URL parameters and routes to the Asset Dashboard.

### B. Global Control Panel (Sidebar)
* **Asset Class Filter**, **Primary Asset Selector**, **Benchmark Overlays**, **Rolling Window**, and **Historical Lookback**.

### C. Tab 1: Asset Dashboard (Pure Analysis)
* Strict historical profiling utilizing **Volatility Regime Shading** on the main chart.

### D. Tab 2: Signals (Execution Layer)
**Objective:** Operationalize analytical data into rules-based directives, explicitly conditioned by current market volatility regime and trend direction.

* **Section A: The Regime-Aware Signal Logic Engine**
    * **Current Regime Display:** Explicitly states the 2D environment (e.g., *"Current Regime: Recovery (High Vol + Up Trend). Prioritizing Volatility-Adjusted Momentum."*)
    * **Hard Veto Condition:** If Volatility > 90th percentile of history, all **Momentum** signals are disabled entirely (Market Breakdown Regime).
    * 🟢 **Momentum State (Prioritizing Hysteresis for Stability):**
        * **Entry:** Segment Rank > 75th Percentile AND Z-Score > 0.5 AND Slope ($\beta$) $\ge$ 0.
        * **Exit (Hysteresis):** Segment Rank < 60th Percentile OR Z-Score < 0.
        * *Regime Modifier:* If triggered during a High Vol Regime, downgrade badge to 🟡 *Weak Momentum (Whipsaw Risk)*.
    * 🔴 **Reversal Zone (Prioritized in High Vol):**
        * Segment Rank < 25th Percentile, Z-Score < -1.0.
        * **[NEW] Slope ($\beta$) > 0** (Ensures the "falling knife" has stopped dropping and the first derivative has turned positive).
        * *Regime Modifier:* If triggered during a High Vol Regime, upgrade badge to 🟢 *Strong Reversal*.

* **Section B: The Action Grid (Flow vs. Reserves)**
    * SIP Flow Multipliers and Dry Powder Deployment triggers.

---

## 7. Automated Testing Specification (`pytest`)
* **Scenario 1–6:** Existing matrix broadcasting, Numba fallback, matrix alignment, holiday integrity, and UI state isolation tests.
* **Scenario 7: Falling Knife Guardrail (Unit Test) [NEW]:** Pass a synthetic array where an asset's price plummets to a Z-score of -2.0 but continues to drop (Slope < 0). Assert that the Reversal Signal does **not** trigger until the slope turns positive.
* **Scenario 8: Regime-Conditioned Downgrades (Unit Test):** Pass a synthetic array with extreme rolling returns (Rank 90th percentile) but massive expanding volatility (Vol > 80th percentile). Assert that the Momentum Signal is correctly downgraded due to the High Volatility regime.
* **Scenario 9: Hard Veto Execution (Unit Test) [NEW]:** Pass a synthetic price array with volatility exceeding the 90th percentile. Assert that all Momentum signals are disabled entirely, regardless of trend or rank strength.

---

## 9. Changelog
* v1.3.0: Initial phase 1.3 architecture definition.
* v1.3.1: Complete instantiation of the Phase 1.3 Dashboard.
* v1.3.2: Expanded asset universe to 21+ liquid Indian Indices and ETFs.
* v1.3.3: Added Spike Filtering specification.
* v1.4.0: Major refactor splitting UI into MVC-style Tab routing. Added Execution Layer.
* v1.5.0: Integrated 2D Market Regimes (Volatility + Trend) and Signal Directionality via Linear Regression Slope ($\beta$).
* v1.6.0: [Current] Transitioned to production-grade signal engine. Added OLS Regression Slope for trend direction, state-machine Hysteresis for signal stability, and a Cross-Sectional Screener with drill-down URL routing. Added Scenario 9 (Hard Veto) to testing suite.
