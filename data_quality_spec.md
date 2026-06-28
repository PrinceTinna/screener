# Data Quality & Sanity Validation Specification

This document details the proposed architectural specification for the **Data Quality & Sanity Validation Framework** in QuantPro. It defines the validation rules, latency guards, corporate event checks for stock splits, and signal-specific sanity checks to prevent false buy/sell signals.

---

## 1. Latency & Stale Data Guards (Data Freshness)

To ensure the dashboard is never analyzing stale or outdated data, the pipeline will execute a freshness validation check on every load.

*   **Rule:** For every active asset in `universe.json`, check the date of the latest row in the price cache.
*   **Threshold:** If `latest_cached_date < current_date - 3 business days`, the asset is marked as **Stale**.
*   **Dashboard Action:** Renders a warning alert at the top of the screen:
    > 🔴 **Data Sync Alert:** The price cache for **{ticker}** is outdated (last updated on {date}). Signals and CAGR metrics may be lagging. Please run the sync scheduler.

---

## 2. Intraday Bar Sanity Checks (Data Feed Corruption)

To protect Numba-compiled rolling CAGR matrices from divide-by-zero errors or mathematical distortions, every ingested daily price row must satisfy the following bounds:

| Check Type | Formula / Condition | Failure Action |
| :--- | :--- | :--- |
| **Negative/Zero Prices** | $Open \le 0$ or $High \le 0$ or $Low \le 0$ or $Close \le 0$ | Mask row as `NaN`, raise data warning |
| **High Boundary** | $High < Open$ or $High < Close$ | Set $High = \max(Open, Close, High)$, log warning |
| **Low Boundary** | $Low > Open$ or $Low > Close$ | Set $Low = \min(Open, Close, Low)$, log warning |
| **Volume Check** | $Volume < 0$ | Set $Volume = 0$ |

---

## 3. High-Fidelity Split Verification (Corporate Action Cross-Check)

Our current split detection checks for daily price drops $>35\%$. To prevent false positives (where a real market crash is incorrectly classified as a split and divided retrospectively), we will implement a **Dual-Gate Verification Layer**:

### Gate A: Index-Reference Cross-Check (Offline & Self-Consistent)
*   **Logic:** If the tracker ETF (e.g. `BANKNIFTY1.NS`) drops by $>35\%$ on Day $T$, query the daily price change of its underlying benchmark index (e.g. `^NSEBANK`) on Day $T$.
*   **Verification:** If the benchmark index daily price change is within normal bounds (e.g. drops by less than $5\%$), then the ETF's price drop is **guaranteed** to be an unadjusted denomination split, not a market crash.
*   **Execution:** Safe to automatically apply the split division.

### Gate B: Corporate Actions API Verification (Online)
*   **Logic:** Query yfinance's official corporate actions endpoint: `yf.Ticker(ticker).splits`.
*   **Verification:** Check if there is an entry in the corporate actions split table within $\pm 3$ business days of the detected drop date.
*   **Execution:** If a matching split date is found, apply the exact split ratio from the table instead of the estimated ratio. If no split is found in the table, raise a `🔴 High-Risk Price Discrepancy` alert in the dashboard and do NOT adjust the history.

---

## 4. Signal-Specific Validation Guards

To ensure that our **Momentum** and **Bubble** signals are completely reliable and free from statistical anomalies, we will implement the following mathematical constraints:

### A. Minimum History Guard (Momentum False-Positive Protection)
*   **Vulnerability:** Newly launched ETFs with short histories (e.g., 30 trading days) can display extreme short-term trend volatility. Scaling this short window to a 1-year annualized CAGR creates massive false-positive momentum buy signals.
*   **Check:** An asset must have a minimum of **252 trading days (~1 year)** of continuous historical price data before any momentum indicator or execution signal is allowed to compute. If history is shorter, the signal is locked to `🟡 Insufficient History`.

### B. Bubble Detection Sample-Size Guard (Bubble False-Alarm Protection)
*   **Vulnerability:** Bubble detection uses a $Z$-score ($Z_{bubble}$) based on the standard deviation of detrended rolling prices. If the asset has a very short history, the rolling standard deviation is extremely small and unstable, causing standard price fluctuations to register as extreme $+3\sigma$ bubble risks.
*   **Check:** Bubble metrics require a minimum of **1,000 trading days (~4 years)** of price history to guarantee a robust historical baseline. If history is less than 1,000 days:
    1. The $Z_{bubble}$ metric is masked to `NaN`.
    2. Bubble alerts/vetoes are disabled for that asset to prevent false-alarm blocks.
    3. The signal display is marked as **`Insufficient History`**.

### C. Fat-Tail Outlier Cap (Data Gap Protection)
*   **Vulnerability:** Missing data bars in history can create sudden vertical gaps in price series, causing rolling CAGR values to spike to unrealistic numbers (e.g., $>300\%$ annualized).
*   **Check:** If an asset's rolling 1-year CAGR exceeds **$150\%$**, the engine flags a validation check:
    *   Verify the percentage of missing bars in the rolling window. If missing bars $>5\%$, downgrade the signal to `🔴 High Uncertainty (Data Gap)` and flag the ticker in the Screener table.

### D. Cross-Sectional Synchronization Guard (Timing Error Protection)
*   **Vulnerability:** Comparing return metrics cross-sectionally (e.g. ranking assets for factor signals) requires time-synchronous inputs. If one asset's data is delayed (e.g. fails to update and stops at Thursday) while others are updated (Friday), we will compare lagged returns with active returns, corrupting the rankings.
*   **Check:** Before running any cross-sectional percentile ranking, the data pipeline must verify that the latest date row of every active ticker is synchronized to the exact same trading day. If any ticker is lagging, it is excluded from the active cross-sectional rank slice to prevent timing errors.

### E. Live Valuation Outlier Threshold (Stale Fundamentals Protection)
*   **Vulnerability:** Live P/E figures pulled from third-party sources like yfinance can sometimes experience reporting anomalies (e.g. listing a P/E of `999x` due to temporary corporate accounting delays).
*   **Check:** If the live P/E fetched from yfinance deviates from the asset's 3-year historical average P/E by more than **$5\sigma$ (standard deviations)**, the check flags it as a valuation outlier. The live P/E is ignored, and the system falls back to the index simulated P/E, preventing stale or corrupted metrics from blocking execution signals.

---

## 5. Operational Optimizations & Caching

To reduce Yahoo Finance API rate-limiting and session blockages:
*   **Cache Extension:** Update the time-to-live (TTL) of the dynamic P/E validator cache from `1 hour` (`ttl=3600`) to **`24 hours`** (`ttl=86400`).
*   **Automatic Flush:** Add a sidebar button (`🔄 Force Refetch Live Data`) to manually clear Streamlit's cache and pull fresh metrics when needed.
