# QuantPro — Index Intelligence Dashboard

QuantPro is a high-fidelity rolling return analysis engine and multi-asset tactical allocation screener designed for tracking the Indian market (NSE/BSE) and broad global indexes. 

Under the hood, the backend functions as an N-dimensional matrix processing engine powered by `vectorbt` and Numba. It processes the entire global multi-asset universe simultaneously using vectorised matrix broadcasting to deliver institutional-grade speed and latency-free calculations.

---

## 🚀 Key Features

*   **Tactical Allocation Engine:** Classifies market regimes and generates execution signals (`🟢 Momentum`, `🔴 Reversal`, `🟡 Neutral`) stabilized by OLS regression slope and hysteresis state machines
*   **Dual-Gate Split Ingestion:** Advanced ingestion adjuster that checks Yahoo Finance corporate action tables and cross-validates large price moves against benchmark reference indices before applying corrections
*   **Global Discovery Matrix:** Cross-sectional screener to group, rank, and compare sectors, smart-beta factors, commodities, and fixed income in real time
*   **Safety Guardrails:** Robust data quality checks locking signals under data anomalies (`🟡 Insufficient History`, `🔴 High Uncertainty (Data Gap)`, `Exclude Stale Assets`)
*   **Valuation Protections:** Valuation Gate mechanism which flags and de-escalates momentum triggers when P/E ratios cross into the $\ge$ 90th historical percentile

---

## 📁 Repository Structure

```
├── config/              # Universe definitions and Streamlit settings
├── core/                # Z-score, trend, and statistical engine mathematics
├── data/                # Ingestion fetchers, pipeline, and fundamentals seeders
├── strategies/          # Allocation multipliers and dry powder triggers
├── tests/               # Automated unit test suite (pipeline, math, signals)
├── ui/                  # Streamlit application layouts and metric guide views
└── README.md            # Repository overview
```

---

## 🛠️ Quick Start

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the local development server:**
    ```bash
    python3 -m streamlit run ui/app.py
    ```

3.  **Run the verification test suite:**
    ```bash
    python3 -m pytest
    ```
