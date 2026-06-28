import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from config.settings import CACHE_DIR, CONFIG_DIR, FETCH_START_YEAR

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.universe_path = CONFIG_DIR / "universe.json"
        with open(self.universe_path, 'r') as f:
            self.universe = json.load(f)
            
    def _adjust_unadjusted_splits(self, df):
        """
        Detects and corrects unadjusted stock splits in historical prices (where yfinance fails to adjust history).
        Uses daily returns to detect vertical drops of >35% on a single day, solves for the ratio,
        and divides all historical prices before that date by the split ratio.
        """
        if df.empty or 'Close' not in df.columns:
            return df
            
        close = df['Close']
        returns = close.pct_change()
        
        # Split dates are characterized by daily drops of >35%
        split_dates = returns[returns < -0.35].index
        
        for s_date in split_dates:
            idx = close.index.get_loc(s_date)
            if idx > 0:
                prev_price = close.iloc[idx - 1]
                curr_price = close.iloc[idx]
                estimated_ratio = prev_price / curr_price
                
                # Match to nearest standard split ratio (2, 3, 4, 5, 10, 20)
                target_ratios = [2, 3, 4, 5, 10, 20]
                split_ratio = min(target_ratios, key=lambda x: abs(x - estimated_ratio))
                
                # Verify that it is a valid split ratio (within 15% of estimated)
                if abs(estimated_ratio - split_ratio) / split_ratio < 0.15:
                    logger.warning(f"🛡️ Data Ingestion: Detected unadjusted stock split of {split_ratio}:1 on {s_date.date()} for asset. Adjusting history prior to this date.")
                    price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                    for col in price_cols:
                        if col in df.columns:
                            df.loc[df.index < s_date, col] = df.loc[df.index < s_date, col] / split_ratio
                            
        return df

    def fetch_all(self, end_date=None):
        """Downloads data for the entire universe with OHLC context and caching."""
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        start_date = f"{FETCH_START_YEAR}-01-01"
        all_data = {}
        
        for ticker, metadata in self.universe.items():
            cache_file = CACHE_DIR / f"{ticker}_raw.parquet"
            # In a real system, we'd check cache here, but for now we fetch to generate baseline
            logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
            
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue
                
                # yFinance sometimes returns multi-index columns for single tickers in newer versions
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                # Correct unadjusted splits in yfinance download stream
                df = self._adjust_unadjusted_splits(df)
                    
                # We save the full OHLCV context as requested
                cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                if 'Adj Close' in df.columns:
                    cols_to_keep.append('Adj Close')
                    
                df = df[[c for c in cols_to_keep if c in df.columns]]
                
                # Save purely raw data to cache
                df.to_parquet(cache_file)
                
                # --- Sync fundamentals ---
                self.update_fundamentals_cache(ticker, df)
                
                all_data[ticker] = df
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                
        return all_data

    def update_fundamentals_cache(self, ticker, df_price):
        """
        Updates the fundamentals cache parquet for the given ticker.
        Appends new dates by either fetching current yFinance fundamentals or using extrapolation fallbacks.
        """
        fund_file = CACHE_DIR / f"{ticker}_fundamentals.parquet"
        meta = self.universe.get(ticker, {})
        broad_class = meta.get("class", "Other").split(" - ")[0]
        
        # 1. Non-equity assets get NaNs (commodities, cash)
        if broad_class in ("Commodities", "Fixed Income", "Cash Equivalent"):
            df_nan = pd.DataFrame({
                "PE": np.nan,
                "EPS": np.nan
            }, index=df_price.index)
            df_nan.to_parquet(fund_file)
            return

        # 2. Check if fundamentals file exists, else seed it
        if not fund_file.exists():
            # Delay import to prevent circular dependency
            from data.fundamentals_seed import generate_fundamental_history, PE_BENCHMARKS, ETF_BENCHMARK_MAP
            config = PE_BENCHMARKS.get(ticker)
            if config is None and ticker in ETF_BENCHMARK_MAP:
                config = PE_BENCHMARKS.get(ETF_BENCHMARK_MAP[ticker])
            if config is None:
                config = {"avg_pe": 20.0, "min_pe": 10.0, "max_pe": 40.0, "growth_rate": 0.11}
            df_fund = generate_fundamental_history(ticker, df_price, config)
            df_fund.to_parquet(fund_file)
            return
            
        # 3. If it exists, read it and check if we need to append new rows
        df_fund = pd.read_parquet(fund_file)
        missing_dates = df_price.index.difference(df_fund.index)
        
        if len(missing_dates) > 0:
            logger.info(f"Syncing {len(missing_dates)} missing fundamental records for {ticker}")
            
            # Fetch current yfinance stats as incremental updates
            try:
                t_obj = yf.Ticker(ticker)
                info = t_obj.info
                current_pe = info.get("trailingPE")
                current_eps = info.get("trailingEps")
            except Exception as e:
                logger.warning(f"Failed to fetch live yfinance fundamentals for {ticker}: {e}")
                current_pe = None
                current_eps = None
                
            new_records = []
            
            # Last available valid PE and EPS to extrapolate from
            last_valid_eps = df_fund["EPS"].dropna().iloc[-1] if not df_fund["EPS"].dropna().empty else 1.0
            last_valid_pe = df_fund["PE"].dropna().iloc[-1] if not df_fund["PE"].dropna().empty else 20.0
            
            # Assume 12% default annual growth rate for extrapolation
            daily_growth = (1 + 0.12) ** (1 / 252)
            
            for date in missing_dates:
                close_series = df_price.loc[date, "Close"]
                close_val = close_series.iloc[-1] if isinstance(close_series, pd.Series) else close_series
                
                # Extrapolate EPS
                last_valid_eps *= daily_growth
                
                # If yfinance has current metrics, use them as anchor, else use derived
                pe = current_pe if current_pe is not None else (close_val / last_valid_eps)
                eps = current_eps if current_eps is not None else last_valid_eps
                
                # Double check PE boundaries to prevent wild spikes
                if not pd.isna(pe):
                    pe = np.clip(pe, 5.0, 60.0)
                    eps = close_val / pe
                    
                new_records.append({
                    "Date": date,
                    "PE": pe,
                    "EPS": eps
                })
                
            df_new = pd.DataFrame(new_records).set_index("Date")
            df_fund = pd.concat([df_fund, df_new])
            df_fund = df_fund[~df_fund.index.duplicated(keep="last")].sort_index()
            df_fund.to_parquet(fund_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = DataFetcher()
    fetcher.fetch_all()
