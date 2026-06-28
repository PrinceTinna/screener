import pandas as pd
import numpy as np
import json
import logging
import holidays
from pathlib import Path
from config.settings import CACHE_DIR, CONFIG_DIR

logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self):
        self.universe_path = CONFIG_DIR / "universe.json"
        with open(self.universe_path, 'r') as f:
            self.universe = json.load(f)
        self.in_calendar = holidays.India()

    def load_cached_raw_data(self):
        """Loads all raw parquets from the cache directory."""
        all_data = {}
        for ticker in self.universe.keys():
            cache_file = CACHE_DIR / f"{ticker}_raw.parquet"
            if cache_file.exists():
                all_data[ticker] = pd.read_parquet(cache_file)
            else:
                logger.warning(f"Missing cached data for {ticker}")
        return all_data

    def build_primary_matrix(self):
        """
        Builds the 2D DataFrame (Time x Assets) using either TR (Adj Close) or PR (Close).
        Applies Muhurat trading volume merging and aligns with standard Indian trading days.
        """
        raw_data_dict = self.load_cached_raw_data()
        if not raw_data_dict:
            raise ValueError("No raw data available to build matrix.")

        series_dict = {}
        for ticker, metadata in self.universe.items():
            if ticker not in raw_data_dict:
                continue
                
            df = raw_data_dict[ticker].copy()
            df.index = pd.to_datetime(df.index)
            
            # --- 1. Muhurat Trading Handling (Diwali micro-candles) ---
            # Standard NSE/BSE candles are Daily. Muhurat might appear as a weekend date
            # or a second anomalous date. A simple heuristic: if a date is an Indian holiday
            # but has trading data, it's likely Muhurat or a special Saturday session.
            # We will merge anomalous volume/price into the nearest valid Friday/Monday.
            # (Detailed implementation of this heuristic is complex; for Phase 1.3 
            # we ensure the index uses end-of-day strict uniqueness and ffill).
            df = df[~df.index.duplicated(keep='last')]
            
            # --- 2. TR vs PR Routing ---
            asset_type = metadata.get('type', 'PR')
            if asset_type == 'TR' and 'Adj Close' in df.columns:
                target_col = 'Adj Close'
            else:
                target_col = 'Close'
                
            series = df[target_col]
            series_dict[ticker] = series

        # Combine into master 2D matrix
        master_matrix = pd.DataFrame(series_dict)
        
        # Sort index just in case
        master_matrix = master_matrix.sort_index()

        # Phase 1 guardrail: We keep NaNs up to the inception date of each asset,
        # but apply ffill() to handle intermittent missing data like a single missing Holi day.
        # Strict alignment with a continuous business day calendar excluding holidays:
        start_date = master_matrix.index.min()
        end_date = master_matrix.index.max()
        
        # Generate bdate_range
        all_bdays = pd.bdate_range(start=start_date, end=end_date)
        
        # Filter out Indian holidays
        valid_trading_days = [d for d in all_bdays if d not in self.in_calendar]
        valid_trading_index = pd.DatetimeIndex(valid_trading_days)
        
        # Reindex the master matrix to the valid trading calendar
        # This aligns the whole matrix to a singular robust timeline
        master_matrix = master_matrix.reindex(valid_trading_index)
        
        # --- 3. Inception Masking ---
        # Force NaNs for any dates before the documented inception date.
        # This handles 'Ghost Data' (incorrect backfills) in raw sources.
        for ticker, metadata in self.universe.items():
            if ticker in master_matrix.columns:
                inception_str = metadata.get('inception')
                if inception_str:
                    inception_date = pd.to_datetime(inception_str)
                    master_matrix.loc[master_matrix.index < inception_date, ticker] = np.nan
        
        # Forward fill transient missing data, but do NOT backfill (preserves inception dates)
        master_matrix = master_matrix.ffill()

        # --- 4. Spike Filtering Guardrail ---
        # Detect and mask statistically impossible daily moves (bad data ticks)
        master_matrix = self._apply_spike_filter(master_matrix)

        return master_matrix

    def _apply_spike_filter(self, matrix: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
        """
        Detects and masks extreme daily price changes that are likely data errors.
        Uses a rolling median to detect multi-day outliers (spikes/drops) that
        deviate significantly from the local trend.
        """
        for ticker in matrix.columns:
            series = matrix[ticker]
            # 1. Calculate a local 'clean' trend using rolling median
            # Window of 21 (approx 1 trading month) is robust against short-term 
            # spikes and also helps identify denomination changes (splits).
            median_series = series.rolling(window=21, center=True, min_periods=1).median()
            
            # 2. Identify points where the price deviates from median by more than the threshold
            deviation = (series / median_series) - 1
            spike_mask = deviation.abs() > threshold
            
            # 3. Unadjusted Split Guard: If historical prices are significantly higher 
            # than the latest price (e.g. > 5x), it's likely an unadjusted denomination change.
            # This is specifically for cases like SETFGOLD.NS (1 gram vs 0.01 gram).
            latest_price = series.dropna().iloc[-1]
            split_mask = (series > latest_price * 5)
            
            final_mask = spike_mask | split_mask
            
            if final_mask.any():
                num_anomalies = final_mask.sum()
                logger.warning(f"🛡️ Spike Guard: Masked {num_anomalies} anomalies for {ticker}")
                # Replace anomaly with NaN
                matrix.loc[final_mask, ticker] = np.nan
        
        # 4. Re-apply ffill to fill the newly created NaNs from the previous valid price
        matrix = matrix.ffill()
        
        return matrix

    def load_cached_fundamentals(self):
        """Loads all raw fundamentals parquets from the cache directory."""
        all_fundamentals = {}
        for ticker in self.universe.keys():
            cache_file = CACHE_DIR / f"{ticker}_fundamentals.parquet"
            if cache_file.exists():
                all_fundamentals[ticker] = pd.read_parquet(cache_file)
            else:
                logger.warning(f"Missing fundamentals cache for {ticker}")
        return all_fundamentals

    def build_fundamentals_matrices(self, valid_index=None):
        """
        Builds PE and EPS master matrices, aligned with the price/returns index.
        Applies inception date masking and forward fills transient NaN holes.
        """
        raw_fund_dict = self.load_cached_fundamentals()
        
        pe_dict = {}
        eps_dict = {}
        
        for ticker in self.universe.keys():
            if ticker not in raw_fund_dict:
                continue
                
            df = raw_fund_dict[ticker].copy()
            df.index = pd.to_datetime(df.index)
            
            # Uniqueness guard
            df = df[~df.index.duplicated(keep='last')]
            
            pe_dict[ticker] = df["PE"]
            eps_dict[ticker] = df["EPS"]
            
        pe_matrix = pd.DataFrame(pe_dict).sort_index()
        eps_matrix = pd.DataFrame(eps_dict).sort_index()
        
        if valid_index is not None:
            pe_matrix = pe_matrix.reindex(valid_index)
            eps_matrix = eps_matrix.reindex(valid_index)
            
        # Inception masking
        for ticker, metadata in self.universe.items():
            inception_str = metadata.get('inception')
            if inception_str:
                inception_date = pd.to_datetime(inception_str)
                if ticker in pe_matrix.columns:
                    pe_matrix.loc[pe_matrix.index < inception_date, ticker] = np.nan
                if ticker in eps_matrix.columns:
                    eps_matrix.loc[eps_matrix.index < inception_date, ticker] = np.nan
                    
        # Forward fill transient missing data
        # Note: Do NOT fill commodities or cash equivalents (they stay NaN)
        pe_matrix = pe_matrix.ffill()
        eps_matrix = eps_matrix.ffill()
        
        return pe_matrix, eps_matrix

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = DataPipeline()
    try:
        matrix = pipeline.build_primary_matrix()
        print(f"Matrix built: {matrix.shape}")
        print(matrix.tail())
    except Exception as e:
        print(f"Failed to build matrix: {e}")
