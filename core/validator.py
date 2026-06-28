import pandas as pd
import yfinance as yf
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# We map core simulated benchmark tickers to liquid ETFs that have yfinance P/Es
VAL_CROSS_CHECK_MAP = {
    "^NSEI": "NIFTYBEES.NS",
    "^NSEBANK": "BANKBEES.NS",
    "^GSPC": "SPY",
    "^NDX": "QQQ",
    "EEM": "EEM"
}

# We map tracker ETFs to their corresponding underlying benchmark index
ETF_BENCHMARK_MAP = {
    "JUNIORBEES.NS": "^NSEI",
    "MID150BEES.NS": "^NSEI",
    "0P0001NJAX.BO": "^NSEI",
    "PSUBNKBEES.NS": "^NSEBANK",
    "INFRABEES.NS": "^NSEI",
    "MOM100.NS": "^NSEI",
    "LOWVOL.NS": "^NSEI"
}

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
                
    # Layer 2: Internal ETF-to-Benchmark index consistency check (PE & EPS Growth)
    for etf_ticker, bench_ticker in ETF_BENCHMARK_MAP.items():
        if etf_ticker in pe_matrix.columns and bench_ticker in pe_matrix.columns:
            # P/E comparison
            etf_pe = pe_matrix[etf_ticker].dropna().iloc[-1]
            bench_pe = pe_matrix[bench_ticker].dropna().iloc[-1]
            if not pd.isna(etf_pe) and not pd.isna(bench_pe) and etf_pe > 0 and bench_pe > 0:
                pe_variance = abs(etf_pe - bench_pe) / bench_pe
                if pe_variance > threshold:
                    alerts.append({
                        "ticker": etf_ticker,
                        "sim_pe": etf_pe,
                        "live_pe": bench_pe,
                        "type": "PE_Internal",
                        "message": (
                            f"⚠️ **Valuation Inconsistency:** Simulated P/E for ETF **{etf_ticker}** ({etf_pe:.1f}x) "
                            f"deviates from its underlying benchmark index **{bench_ticker}** ({bench_pe:.1f}x) by **{pe_variance*100:.1f}%**."
                        )
                    })
                    
        if eps_matrix is not None and etf_ticker in eps_matrix.columns and bench_ticker in eps_matrix.columns:
            # YoY EPS growth comparison
            etf_eps_series = eps_matrix[etf_ticker].dropna()
            bench_eps_series = eps_matrix[bench_ticker].dropna()
            
            if len(etf_eps_series) >= 253 and len(bench_eps_series) >= 253:
                etf_growth = (etf_eps_series.iloc[-1] / etf_eps_series.iloc[-252]) - 1.0
                bench_growth = (bench_eps_series.iloc[-1] / bench_eps_series.iloc[-252]) - 1.0
                
                # Check for absolute variance difference
                growth_variance = abs(etf_growth - bench_growth)
                if growth_variance > threshold:
                    note = ""
                    if etf_ticker in ("MON100.NS", "QQQ", "MOM100.NS"):
                        note = " (Note: This is common for international or momentum ETFs subject to SEBI/RBI investment limit restrictions or tracking error premium expansions on local exchanges)"
                    
                    alerts.append({
                        "ticker": f"{etf_ticker}_EPS",
                        "sim_pe": etf_growth * 100,
                        "live_pe": bench_growth * 100,
                        "type": "EPS_Internal",
                        "message": (
                            f"⚠️ **Earnings Growth Drift:** YoY EPS growth for ETF **{etf_ticker}** ({etf_growth*100:.1f}%) "
                            f"deviates from its benchmark index **{bench_ticker}** ({bench_growth*100:.1f}%) by **{growth_variance*100:.1f} percentage points**{note}."
                        )
                    })
                    
    return alerts
