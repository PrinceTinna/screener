import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# ── Matrix Validation (migrated from core/validators.py) ──────────────────────

def validate_matrix_shape(matrix: pd.DataFrame, universe: dict) -> bool:
    """
    Ensures the matrix contains all expected assets from the universe.
    """
    expected_tickers = set(universe.keys())
    actual_tickers = set(matrix.columns)
    
    missing = expected_tickers - actual_tickers
    if missing:
        raise ValueError(f"Matrix validation failed. Missing tickers: {missing}")
    return True

def validate_inception_alignment(matrix: pd.DataFrame, universe: dict):
    """
    Validates that no data structurally exists before the documented inception date.
    (Prevents 'ghost returns' via backfilling errors).
    """
    for ticker, meta in universe.items():
        if ticker in matrix.columns:
            inception_str = meta.get('inception', None)
            if inception_str:
                inception_date = pd.to_datetime(inception_str)
                # Check if there are non-NaN prices before inception date
                pre_inception_mask = matrix.index < inception_date
                pre_inception_data = matrix.loc[pre_inception_mask, ticker]
                if not pre_inception_data.isna().all():
                    raise ValueError(f"Ghost Data Error: {ticker} contains active price data before inception ({inception_date.date()})")
    
    return True

# ── Valuation Cross-Check ─────────────────────────────────────────────────────

# We map core simulated benchmark tickers to liquid ETFs that have yfinance P/Es
VAL_CROSS_CHECK_MAP = {
    "^NSEI": "NIFTYBEES.NS",
    "^NSEBANK": "BANKBEES.NS",
    "^GSPC": "SPY",
    "^NDX": "QQQ",
    "EEM": "EEM"
}

# NOTE: ETF_BENCHMARK_MAP was removed. It previously mapped ETFs to proxy
# indices for fundamental data, but all mappings were inaccurate proxies.
# See data/fundamentals_seed.py for the get_fundamental_tier() system.

@st.cache_data(ttl=86400)  # Cache yfinance lookups for 24 hours to prevent API throttling
def fetch_live_etf_pes():
    """Fetches live P/E values from yfinance for validator benchmarks."""
    live_pes = {}
    for index_ticker, etf_ticker in VAL_CROSS_CHECK_MAP.items():
        try:
            t = yf.Ticker(etf_ticker)
            info = t.info
            pe = info.get("trailingPE") or info.get("forwardPE")
            if pe:
                live_pes[index_ticker] = float(pe)
        except Exception as e:
            logger.warning(f"Failed to fetch cross-check P/E for {etf_ticker}: {e}")
    return live_pes

def run_valuation_cross_check(pe_matrix: pd.DataFrame, eps_matrix: pd.DataFrame, threshold: float = 0.20):
    """
    Compares database simulated P/E and EPS values against live references and benchmark indices.
    Returns a list of alerts if variance exceeds the threshold.
    """
    if pe_matrix is None or pe_matrix.empty:
        return []
        
    alerts = []
    
    # Layer 1: External live P/E cross check
    live_pes = fetch_live_etf_pes()
    for index_ticker, live_pe in live_pes.items():
        if index_ticker in pe_matrix.columns:
            sim_pe = pe_matrix[index_ticker].dropna().iloc[-1]
            if pd.isna(sim_pe) or sim_pe <= 0:
                continue
                
            # Check E: Live Valuation Outlier Threshold (5-sigma check over last 3 years)
            hist_pe = pe_matrix[index_ticker].dropna().tail(756) # 3 years @ 252 days/yr
            if len(hist_pe) > 30:
                mean_pe = hist_pe.mean()
                std_pe = hist_pe.std()
                if std_pe > 0 and abs(live_pe - mean_pe) > 5 * std_pe:
                    logger.warning(
                        f"⚠️ Live Valuation Outlier: Live P/E for {index_ticker} is {live_pe:.1f}x, "
                        f"deviating by >5σ from 3y mean ({mean_pe:.1f}x ± {std_pe:.1f}x). Ignoring live reference."
                    )
                    continue
                    
            variance = abs(sim_pe - live_pe) / live_pe
            if variance > threshold:
                alerts.append({
                    "ticker": index_ticker,
                    "sim_pe": sim_pe,
                    "live_pe": live_pe,
                    "type": "PE_External",
                    "message": (
                        f"⚠️ **Valuation Drift Alert:** Simulated P/E for **{index_ticker}** ({sim_pe:.1f}x) "
                        f"deviates from live ETF market reference ({live_pe:.1f}x) by **{variance*100:.1f}%**."
                    )
                })
                
    # Layer 2 was removed: Previously compared ETF fundamentals to their proxy
    # benchmark (ETF_BENCHMARK_MAP), but all mappings were inaccurate proxies.
    # ETFs now have NaN fundamentals (Tier 3), so this check is not applicable.
    # When real fundamental data sources are integrated, add a new Layer 2
    # that validates against live API data instead.
                    
    return alerts

