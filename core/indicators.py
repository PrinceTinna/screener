import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit
from config.settings import TRADING_DAYS_PER_YEAR

@njit(cache=True)
def fast_percentiles_2d(matrix):
    """
    Computes cross-sectional (row-wise) percentiles across the 2D matrix.
    Skips NaNs efficiently. Returns dict of percentile arrays.
    """
    rows, cols = matrix.shape
    p10 = np.full(rows, np.nan)
    p25 = np.full(rows, np.nan)
    p50 = np.full(rows, np.nan)
    p75 = np.full(rows, np.nan)
    p90 = np.full(rows, np.nan)
    
    for i in range(rows):
        row_data = matrix[i, :]
        # Filter NaNs for the cross section
        valid_data = row_data[~np.isnan(row_data)]
        if len(valid_data) > 0:
            p10[i] = np.percentile(valid_data, 10)
            p25[i] = np.percentile(valid_data, 25)
            p50[i] = np.percentile(valid_data, 50)
            p75[i] = np.percentile(valid_data, 75)
            p90[i] = np.percentile(valid_data, 90)
            
    return p10, p25, p50, p75, p90

class MathEngine:
    def __init__(self, master_matrix):
        """
        Expects a clean 2D pd.DataFrame containing adjusted prices.
        """
        self.matrix = master_matrix

    def calculate_rolling_returns(self, window_days: int) -> pd.DataFrame:
        """
        Calculates R_{t,w} simultaneously using vectorbt broadcast.
        """
        # Vectorized rolling return: (Price_t / Price_{t-w}) - 1
        return self.matrix.pct_change(periods=window_days)

    def calculate_cagr(self, rolling_returns: pd.DataFrame, window_days: int) -> pd.DataFrame:
        """
        Converts a rolling return matrix to an annualized CAGR matrix.
        """
        years = window_days / TRADING_DAYS_PER_YEAR
        return (1 + rolling_returns) ** (1 / years) - 1

    def get_cross_sectional_percentiles(self, metric_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Wraps the Numba percentile calculation and returns a DataFrame.
        """
        raw_values = metric_matrix.values
        p10, p25, p50, p75, p90 = fast_percentiles_2d(raw_values)
        
        return pd.DataFrame({
            'P10': p10,
            'P25': p25,
            'Median': p50,
            'P75': p75,
            'P90': p90
        }, index=metric_matrix.index)
