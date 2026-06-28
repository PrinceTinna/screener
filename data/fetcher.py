import yfinance as yf
# Clear yfinance singleton session cache to evict any stale standard requests session
try:
    from yfinance.data import YfData
    YfData._instances.pop(YfData, None)
except Exception:
    pass

import pandas as pd
import numpy as np
import json
import logging
import time
import random
import requests
from datetime import datetime
from pathlib import Path
from config.settings import CACHE_DIR, CONFIG_DIR, FETCH_START_YEAR

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.universe_path = CONFIG_DIR / "universe.json"
        with open(self.universe_path, 'r') as f:
            self.universe = json.load(f)
        
        # Delay imports if needed
            
    def _adjust_unadjusted_splits(self, df, ticker):
        """
        Detects and corrects unadjusted stock splits in historical prices (where yfinance fails to adjust history).
        Uses a dual-gate validation layer:
        Gate A (Index-Reference Cross-Check): Compares drop to tracking index performance.
        Gate B (Corporate Actions table): Validates split date and ratio via yfinance splits API.
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
                
                # --- Gate A: Index-Reference Cross-Check ---
                gate_a_passed = False
                # Split reference map: maps ETFs to a related index for split detection ONLY.
                # If the ETF drops 50% but the reference index didn't, it's likely a split.
                # This is NOT used for fundamentals (PE/EPS) — those use the tier system.
                SPLIT_REFERENCE_MAP = {
                    "JUNIORBEES.NS": "^NSEI",
                    "0P0001NJAX.BO": "^NSEI",
                    "PSUBNKBEES.NS": "^NSEBANK",
                    "INFRABEES.NS": "^NSEI",
                    "MOM100.NS": "^NSEI",
                    "LOWVOL.NS": "^NSEI",
                }
                benchmark = SPLIT_REFERENCE_MAP.get(ticker)
                
                if benchmark:
                    bench_file = CACHE_DIR / f"{benchmark}_raw.parquet"
                    if bench_file.exists():
                        try:
                            bench_df = pd.read_parquet(bench_file)
                            if not bench_df.empty and 'Close' in bench_df.columns:
                                bench_close = bench_df['Close']
                                if s_date in bench_close.index:
                                    bench_idx = bench_close.index.get_loc(s_date)
                                    if bench_idx > 0:
                                        bench_return = bench_close.iloc[bench_idx] / bench_close.iloc[bench_idx - 1] - 1.0
                                        # If the benchmark index dropped by less than 5%, then Gate A passes (it was a split, not a crash)
                                        if bench_return > -0.05:
                                            gate_a_passed = True
                                            logger.info(f"🛡️ Split Gate A Passed: Ticker {ticker} benchmark {benchmark} change on {s_date.date()} was {bench_return:.2%}. Safe to split-adjust.")
                        except Exception as e:
                            logger.warning(f"Error checking benchmark split reference for {ticker} (benchmark {benchmark}): {e}")
                
                # --- Gate B: Corporate Actions API Verification ---
                gate_b_passed = False
                found_ratio = None
                try:
                    splits_series = yf.Ticker(ticker).splits
                    if not splits_series.empty:
                        for split_date, ratio in splits_series.items():
                            # splits indexing timezone may differ, compare dates
                            if abs((pd.to_datetime(split_date).tz_localize(None) - s_date.tz_localize(None)).days) <= 3:
                                gate_b_passed = True
                                found_ratio = float(ratio)
                                logger.info(f"🛡️ Split Gate B Passed: Ticker {ticker} splits record found ratio {found_ratio} near {s_date.date()}.")
                                break
                except Exception as e:
                    logger.warning(f"Could not fetch yfinance splits corporate actions for {ticker}: {e}")
                
                # --- Decision Logic ---
                if gate_b_passed and found_ratio is not None:
                    # Gate B is the highest fidelity source, use its exact ratio
                    split_ratio = found_ratio
                elif gate_a_passed:
                    # Gate A passed (index didn't drop), but Gate B was missing splits entry (typical for Indian ETFs).
                    # Verify estimated ratio is standard:
                    if abs(estimated_ratio - split_ratio) / split_ratio < 0.15:
                        pass # use standard estimated split_ratio
                    else:
                        logger.warning(f"❌ Split adjustment cancelled for {ticker} on {s_date.date()}: Estimated ratio {estimated_ratio:.2f} deviates too far from standard ratios.")
                        continue
                else:
                    # Both gates failed (or Gate A was not available and Gate B had no entry)
                    logger.warning(f"❌ Split adjustment REJECTED/CANCELLED for {ticker} on {s_date.date()}. Both Index-Reference and Splits API Gates failed. Possible actual market crash.")
                    continue
                
                logger.warning(f"🛡️ Data Ingestion: Adjusting history for {ticker} prior to {s_date.date()} by split ratio of {split_ratio:.2f}:1")
                price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
                for col in price_cols:
                    if col in df.columns:
                        df.loc[df.index < s_date, col] = df.loc[df.index < s_date, col] / split_ratio
                            
        return df

    def _sanitize_price_bars(self, df, ticker):
        """
        Validates open-high-low-close boundaries, negative/zero pricing, and volume anomalies.
        Masks corrupted entries to prevent NaN/JIT engine failures.
        """
        if df.empty:
            return df
            
        # 1. Negative/Zero prices check: mask to NaN
        mask_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in mask_cols:
            if col in df.columns:
                invalid_mask = df[col] <= 0
                if invalid_mask.any():
                    logger.warning(f"⚠️ Data Sanity: Found negative/zero values in {col} for {ticker}. Masking to NaN.")
                    df.loc[invalid_mask, col] = np.nan
                    
        # 2. High Boundary: High must be >= Open and >= Close
        if 'High' in df.columns:
            for col in ['Open', 'Close']:
                if col in df.columns:
                    mismatch = df['High'] < df[col]
                    if mismatch.any():
                        logger.warning(f"⚠️ Data Sanity: High < {col} for {ticker}. Adjusting High boundary.")
                        df.loc[mismatch, 'High'] = df.loc[mismatch, [col, 'High']].max(axis=1)
                        
        # 3. Low Boundary: Low must be <= Open and <= Close
        if 'Low' in df.columns:
            for col in ['Open', 'Close']:
                if col in df.columns:
                    mismatch = df['Low'] > df[col]
                    if mismatch.any():
                        logger.warning(f"⚠️ Data Sanity: Low > {col} for {ticker}. Adjusting Low boundary.")
                        df.loc[mismatch, 'Low'] = df.loc[mismatch, [col, 'Low']].min(axis=1)
                        
        # 4. Volume Check: Volume must be >= 0
        if 'Volume' in df.columns:
            invalid_volume = df['Volume'] < 0
            if invalid_volume.any():
                logger.warning(f"⚠️ Data Sanity: Negative volume found for {ticker}. Setting to 0.")
                df.loc[invalid_volume, 'Volume'] = 0.0
                
        return df

    def fetch_all(self, end_date=None):
        """Downloads data for the entire universe with OHLC context and caching."""
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
            
        lock_file = CACHE_DIR / "data.lock"
        
        # Concurrency Lock Mutex
        if lock_file.exists():
            # If the lock file is older than 5 minutes, it's likely a leftover from a crashed run
            if time.time() - lock_file.stat().st_mtime > 300:
                logger.warning("Found a stale lock file. Removing it to proceed.")
                try:
                    lock_file.unlink()
                except Exception:
                    pass
            else:
                logger.warning("Data fetch is currently locked by another process. Waiting...")
                for _ in range(10):
                    time.sleep(1)
                    if not lock_file.exists():
                        break
                else:
                    logger.error("Lock file still present. Exiting to prevent concurrency collision.")
                    return {}

        # Create lock file
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            lock_file.touch()
        except Exception as e:
            logger.warning(f"Could not create lock file: {e}")

        all_data = {}
        
        try:
            for ticker, metadata in self.universe.items():
                cache_file = CACHE_DIR / f"{ticker}_raw.parquet"
                df_cached = None
                start_date = f"{FETCH_START_YEAR}-01-01"
                
                if cache_file.exists():
                    try:
                        df_cached = pd.read_parquet(cache_file)
                        if not df_cached.empty:
                            last_date = df_cached.index.max()
                            # Start fetch 1 day after the last cached date
                            start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    except Exception as cache_err:
                        logger.warning(f"Error reading raw cache for {ticker}, falling back to full fetch: {cache_err}")
                
                # If start_date >= end_date, we don't need to fetch
                if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
                    logger.info(f"Ticker {ticker} is already up to date (Last date: {start_date}). Skipping fetch.")
                    if df_cached is not None:
                        all_data[ticker] = df_cached
                    continue
                    
                # Sleep jitter (Rate Limiting) before making the active network call
                time.sleep(random.uniform(1.0, 2.5))
                logger.info(f"Fetching incremental data for {ticker} from {start_date} to {end_date}")
                
                try:
                    df_new = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date, 
                        progress=False
                    )
                    
                    if df_new.empty:
                        logger.warning(f"No new data returned for {ticker}")
                        if df_cached is not None:
                            all_data[ticker] = df_cached
                        continue
                    
                    # yFinance sometimes returns multi-index columns for single tickers in newer versions
                    if isinstance(df_new.columns, pd.MultiIndex):
                        df_new.columns = df_new.columns.get_level_values(0)
                    
                    # Combine cached and new data
                    if df_cached is not None:
                        df = pd.concat([df_cached, df_new])
                    else:
                        df = df_new
                        
                    df = df[~df.index.duplicated(keep='last')].sort_index()
                        
                    # Correct unadjusted splits in yfinance download stream
                    df = self._adjust_unadjusted_splits(df, ticker)
                    
                    # Sanitize price bars for negative/zero prices and boundary errors
                    df = self._sanitize_price_bars(df, ticker)
                        
                    # We save the full OHLCV context as requested
                    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if 'Adj Close' in df.columns:
                        cols_to_keep.append('Adj Close')
                        
                    df = df[[c for c in cols_to_keep if c in df.columns]]
                    
                    # Save purely raw data to cache
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(cache_file)
                    
                    # --- Sync fundamentals ---
                    self.update_fundamentals_cache(ticker, df)
                    
                    all_data[ticker] = df
                    
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
                    if df_cached is not None:
                        all_data[ticker] = df_cached
        finally:
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except Exception:
                    pass
                    
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
            fund_file.parent.mkdir(parents=True, exist_ok=True)
            df_nan.to_parquet(fund_file)
            return

        # 2. Check if fundamentals file exists, else seed it
        if not fund_file.exists():
            # Delay import to prevent circular dependency
            from data.fundamentals_seed import generate_fundamental_history, PE_BENCHMARKS, get_fundamental_tier
            
            # Classify asset into fundamental data confidence tier
            tier = get_fundamental_tier(ticker, meta.get("class", "Other"))
            
            if tier == 3:
                # Tier 3: No reliable fundamental source → write NaN parquet
                # This covers ETFs (no exact-match benchmark), Smart Beta, etc.
                logger.info(f"Tier 3 asset {ticker}: No fundamental data available, writing NaN")
                df_nan = pd.DataFrame({
                    "PE": np.nan,
                    "EPS": np.nan
                }, index=df_price.index)
                fund_file.parent.mkdir(parents=True, exist_ok=True)
                df_nan.to_parquet(fund_file)
                return
            
            # Tier 1: Has own PE benchmark — generate model-estimated fundamentals
            config = PE_BENCHMARKS.get(ticker)
            if config is None:
                # Safety fallback — should not happen for Tier 1, but guard anyway
                logger.warning(f"Tier 1 asset {ticker} missing PE_BENCHMARKS config, writing NaN")
                df_nan = pd.DataFrame({"PE": np.nan, "EPS": np.nan}, index=df_price.index)
                fund_file.parent.mkdir(parents=True, exist_ok=True)
                df_nan.to_parquet(fund_file)
                return
                
            df_fund = generate_fundamental_history(ticker, df_price, config)
            fund_file.parent.mkdir(parents=True, exist_ok=True)
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
            fund_file.parent.mkdir(parents=True, exist_ok=True)
            df_fund.to_parquet(fund_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = DataFetcher()
    fetcher.fetch_all()
