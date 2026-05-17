import yfinance as yf
import pandas as pd
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
                    
                # We save the full OHLCV context as requested
                cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                if 'Adj Close' in df.columns:
                    cols_to_keep.append('Adj Close')
                    
                df = df[[c for c in cols_to_keep if c in df.columns]]
                
                # Save purely raw data to cache
                df.to_parquet(cache_file)
                all_data[ticker] = df
                
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
                
        return all_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = DataFetcher()
    fetcher.fetch_all()
