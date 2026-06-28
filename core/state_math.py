import numpy as np
import pandas as pd
from numba import njit

def calculate_z_score(series: pd.Series, window: int = 252) -> pd.Series:
    r"""
    Calculates the rolling Z-Score of a series.
    Z = (X - \mu) / \sigma
    """
    rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
    rolling_std = series.rolling(window=window, min_periods=window//2).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    z_score = (series - rolling_mean) / rolling_std
    return z_score

def calculate_trend(series: pd.Series, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """
    Calculates Simple Moving Averages (SMA) for trend detection.
    Returns a DataFrame with 'fast_sma', 'slow_sma', and a boolean 'is_positive_trend'.
    """
    fast_sma = series.rolling(window=fast, min_periods=fast//2).mean()
    slow_sma = series.rolling(window=slow, min_periods=slow//2).mean()
    
    is_positive_trend = fast_sma > slow_sma
    
    return pd.DataFrame({
        'fast_sma': fast_sma,
        'slow_sma': slow_sma,
        'is_positive_trend': is_positive_trend
    })

@njit(cache=True)
def _ols_slope_numba(y):
    """
    Numba-optimized OLS slope calculation for a single window.
    """
    n = len(y)
    x = np.arange(float(n))
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

def calculate_ols_slope(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Calculates the rolling OLS Regression Slope (Beta) of a series.
    Provides a stable, noise-filtered measure of momentum direction.
    """
    return series.rolling(window=window, min_periods=window).apply(_ols_slope_numba, raw=True)

def classify_regime_2d(vol_series: pd.Series, price_series: pd.Series, window: int = 252) -> pd.Series:
    """
    Classifies market regimes based on Volatility (relative to history) and Trend.
    Returns a categorical series of regimes.
    
    Fully vectorized implementation using boolean masks and np.select().
    """
    # 1. Volatility Classification
    # We use rolling quantile of volatility to determine relative state
    vol_q40 = vol_series.rolling(window=window*2, min_periods=window).quantile(0.4)
    vol_q75 = vol_series.rolling(window=window*2, min_periods=window).quantile(0.75)
    vol_q90 = vol_series.rolling(window=window*2, min_periods=window).quantile(0.9)
    
    # 2. Trend Classification (Price relative to SMA)
    sma = price_series.rolling(window=window, min_periods=window//2).mean()
    is_uptrend = (price_series > sma).reindex(vol_series.index, fill_value=False)
    
    # 3. Vectorized boolean masks (all aligned to vol_series.index)
    has_data = vol_series.notna() & vol_q40.notna()
    is_extreme_vol = vol_series > vol_q90
    is_high_vol = vol_series > vol_q75
    is_low_vol = vol_series < vol_q40
    
    # 4. Regime conditions (order matters — checked top-to-bottom)
    conditions = [
        ~has_data,                                # Unknown (NaN)
        has_data & is_extreme_vol,                # Extreme Breakdown
        has_data & is_high_vol & is_uptrend,      # Recovery
        has_data & is_high_vol & ~is_uptrend,     # Panic
        has_data & is_low_vol & is_uptrend,       # Trending Bull
        has_data & is_low_vol & ~is_uptrend,      # Low-Vol Range
        has_data & ~is_low_vol & ~is_high_vol & is_uptrend,   # Neutral Bull
        has_data & ~is_low_vol & ~is_high_vol & ~is_uptrend,  # Neutral Bear
    ]
    
    choices = [
        "Unknown",
        "Extreme Breakdown",
        "Recovery",
        "Panic",
        "Trending Bull",
        "Low-Vol Range",
        "Neutral Bull",
        "Neutral Bear",
    ]
    
    regimes = np.select(conditions, choices, default="Unknown")
    
    return pd.Series(regimes, index=vol_series.index)

def calculate_bubble_z_score(price_series: pd.Series, trend_window: int = 756) -> pd.Series:
    """
    Calculates the detrended price distance Z-Score relative to its long-term baseline.
    D_t = (Price_t - SMA_{t, 756}) / SMA_{t, 756}
    Z_bubble_t = (D_t - expanding_mean(D)) / expanding_std(D)

    Uses an expanding window (not lifetime) to avoid look-ahead bias.
    Each day's Z-score only uses data available up to that point.
    """
    if len(price_series) < trend_window:
        return pd.Series(np.nan, index=price_series.index)
    
    # 1. Calculate long-term trend (e.g. 3Y SMA)
    # Using min_periods=trend_window//2 to allow starting data calculation early but maintaining stability
    sma = price_series.rolling(window=trend_window, min_periods=trend_window//2).mean()
    
    # 2. Distance from trend
    distance = (price_series - sma) / sma
    
    # 3. Standardize using expanding mean and std (no look-ahead bias)
    expanding_mean = distance.expanding(min_periods=trend_window).mean()
    expanding_std = distance.expanding(min_periods=trend_window).std()
    
    # Avoid division by zero
    expanding_std = expanding_std.replace(0, np.nan)
        
    bubble_z = (distance - expanding_mean) / expanding_std
    return bubble_z

def classify_bubble_status(bubble_z: float) -> dict:
    """
    Classifies bubble status and returns label, badge style, emoji, and status string.
    """
    if pd.isna(bubble_z):
        return {
            "status": "Unknown",
            "badge": "info",
            "emoji": "⚪",
            "label": "⚪ Unknown"
        }
    if bubble_z < 1.5:
        return {
            "status": "Normal",
            "badge": "success",
            "emoji": "🟢",
            "label": "🟢 Normal"
        }
    if bubble_z < 2.0:
        return {
            "status": "Extended",
            "badge": "warning",
            "emoji": "🟡",
            "label": "🟡 Extended"
        }
    if bubble_z < 3.0:
        return {
            "status": "2-Sigma Bubble",
            "badge": "warning",
            "emoji": "🟠",
            "label": "🟠 2-Sigma Bubble"
        }
    return {
        "status": "3-Sigma Superbubble",
        "badge": "error",
        "emoji": "🔴",
        "label": "🔴 3-Sigma Superbubble"
    }

def classify_bubble_series(bubble_z_series: pd.Series) -> pd.Series:
    """
    Vectorized classification of bubble status for a series.
    """
    conditions = [
        bubble_z_series.isna(),
        bubble_z_series < 1.5,
        (bubble_z_series >= 1.5) & (bubble_z_series < 2.0),
        (bubble_z_series >= 2.0) & (bubble_z_series < 3.0),
        bubble_z_series >= 3.0
    ]
    choices = [
        "Unknown",
        "Normal",
        "Extended",
        "2-Sigma Bubble",
        "3-Sigma Superbubble"
    ]
    regimes = np.select(conditions, choices, default="Unknown")
    return pd.Series(regimes, index=bubble_z_series.index)
