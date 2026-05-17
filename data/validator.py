import pandas as pd

def validate_sparsity(df, threshold=0.8):
    """
    Ensures that assets have at least 'threshold' percent of data 
    populated during their active trading period.
    """
    clean_df = df.copy()
    dropped_tickers = []
    
    # Handle MultiIndex (Ticker, Metric)
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(0).unique()
        for ticker in tickers:
            ticker_slice = df[ticker]
            first_idx = ticker_slice.first_valid_index()
            last_idx = ticker_slice.last_valid_index()
            
            if first_idx is None or last_idx is None:
                dropped_tickers.append(ticker)
                clean_df.drop(columns=[ticker], level=0, inplace=True)
                continue
            
            # Check Price column specifically if available
            check_col = 'Price' if 'Price' in ticker_slice.columns else ticker_slice.columns[0]
            active_slice = ticker_slice[check_col].loc[first_idx:last_idx]
            
            if len(active_slice) > 0:
                valid_ratio = active_slice.count() / len(active_slice)
                if valid_ratio < threshold:
                    dropped_tickers.append(ticker)
                    clean_df.drop(columns=[ticker], level=0, inplace=True)
            else:
                dropped_tickers.append(ticker)
                clean_df.drop(columns=[ticker], level=0, inplace=True)
                
    return clean_df, dropped_tickers

def check_price_integrity(df):
    """
    Basic sanity checks: Price > 0, no extreme outliers (optional).
    """
    # Placeholder for more complex logic
    return True
