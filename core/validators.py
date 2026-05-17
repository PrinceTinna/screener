import pandas as pd
import numpy as np

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
