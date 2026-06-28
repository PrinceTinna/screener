import numpy as np
import pandas as pd
from numba import njit
import math

@njit(cache=True)
def calculate_cagr_nb(price_today, price_past, window_days):
    """JIT-compatible CAGR calculation."""
    if price_past <= 0:
        return np.nan
    return math.pow((price_today / price_past), (252.0 / window_days)) - 1.0

@njit(cache=True)
def calculate_percentile_nb(current_val, history):
    """JIT-compatible Percentile Rank calculation."""
    if len(history) == 0:
        return np.nan
    smaller_count = 0.0
    for val in history:
        if val < current_val:
            smaller_count += 1.0
    return (smaller_count / len(history)) * 100.0

def calculate_net_median_return(pf, mtf_pct=9.7, tax_pct=20.0, return_breakdown=False):
    """
    Calculates the Net Median Per-Trade Return after deducting pro-rated MTF interest and STCG Tax.
    Note: This is NOT a true IRR (which requires solving for the discount rate that
    makes NPV=0). It is the median of individual trade net returns.
    """
    if pf is None or pf.trades.count().sum() == 0:
        return (0.0, {}) if return_breakdown else 0.0
    
    durations_days = pf.trades.duration.values
    gross_returns = pf.trades.returns.values
    
    # Pro-rate Annual MTF to Trade Duration
    mtf_costs = (mtf_pct / 100.0) * (durations_days / 365.0)
    post_mtf_returns = gross_returns - mtf_costs
    
    # Apply STCG Tax to positive returns
    tax_factor = 1.0 - (tax_pct / 100.0)
    final_returns = np.where(post_mtf_returns > 0, post_mtf_returns * tax_factor, post_mtf_returns)
    
    net_median = np.median(final_returns) if len(final_returns) > 0 else 0.0
    
    if return_breakdown:
        tax_impact = np.median(np.where(post_mtf_returns > 0, post_mtf_returns * (tax_pct/100.0), 0))
        pre_tax_val = net_median + tax_impact
        avg_mtf = np.median(mtf_costs)
        gross_val = pre_tax_val + avg_mtf
        
        breakdown = {
            "Gross Returns (%)": float(gross_val * 100),
            "Avg MTF Interest Drag (%)": float(avg_mtf * 100),
            "Pre-Tax Net Return (%)": float(pre_tax_val * 100),
            "STCG Tax Impact (%)": float(tax_impact * 100),
            "Final Net Median Return (%)": float(net_median * 100)
        }
        return net_median, breakdown

    return net_median
