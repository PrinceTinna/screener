import pytest
import pandas as pd
import numpy as np
from core.validator import validate_matrix_shape, validate_inception_alignment

def test_validate_inception_alignment():
    """Scenario 3: Indian Matrix Alignment (NaN backfill check)"""
    universe = {
        "NIFTYBEES.NS": {"inception": "2001-12-28"},
        "NEW_ETF.NS": {"inception": "2023-01-01"}
    }
    
    # Simulate a matrix where NEW_ETF.NS violates the rule by having data in 2022
    dates = pd.date_range("2022-12-30", "2023-01-05", freq='B')
    
    # Correct case (NaNs before inception)
    matrix_correct = pd.DataFrame({
        "NIFTYBEES.NS": [100.0] * len(dates),
        "NEW_ETF.NS": [np.nan, np.nan, 50.0, 51.0, 52.0]
    }, index=dates)
    
    assert validate_inception_alignment(matrix_correct, universe) == True
    
    # Incorrect case (Ghost data)
    matrix_incorrect = pd.DataFrame({
        "NIFTYBEES.NS": [100.0] * len(dates),
        "NEW_ETF.NS": [49.0, 49.5, 50.0, 51.0, 52.0] # Backfilled!
    }, index=dates)
    
    with pytest.raises(ValueError, match="Ghost Data Error"):
        validate_inception_alignment(matrix_incorrect, universe)

def test_spike_filter():
    """Verifies that the spike filter correctly masks and ffills extreme price anomalies using rolling median."""
    from data.pipeline import DataPipeline
    
    # Mock pipeline (we only need the _apply_spike_filter method)
    with patch.object(DataPipeline, "__init__", return_value=None):
        pipeline = DataPipeline()
        
        # Scenario: Decimal error (100x drop for 2 days and recovery)
        dates = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            "TICKER": [100.0, 100.5, 101.0, 101.5, 1.01, 1.02, 102.0, 102.5, 103.0, 103.5] 
        }, index=dates)
        
        cleaned_df = pipeline._apply_spike_filter(df, threshold=0.30)
        
        # Day 5 (1.01) and Day 6 (1.02) should be masked and ffilled with 101.5
        assert cleaned_df.loc["2024-01-05", "TICKER"] == 101.5
        assert cleaned_df.loc["2024-01-06", "TICKER"] == 101.5
        
        # Surrounding days should remain unchanged
        assert cleaned_df.loc["2024-01-04", "TICKER"] == 101.5
        assert cleaned_df.loc["2024-01-07", "TICKER"] == 102.0

from unittest.mock import patch

def test_fundamentals_pipeline_alignment():
    """Verify that fundamentals matrices align and map properly, handling NaNs safely."""
    from data.pipeline import DataPipeline
    pipeline = DataPipeline()
    
    # Run matrix building
    price_matrix = pipeline.build_primary_matrix()
    pe_matrix, eps_matrix = pipeline.build_fundamentals_matrices(price_matrix.index)
    
    # Assert matrices have correct shape and columns matching the universe
    assert pe_matrix.shape[1] == price_matrix.shape[1]
    assert eps_matrix.shape[1] == price_matrix.shape[1]
    assert pe_matrix.index.equals(price_matrix.index)
    
    # Assert Commodities & Fixed Income have NaN values (NaN safety checks)
    assert pe_matrix["GOLDBEES.NS"].isna().all()
    assert eps_matrix["GOLDBEES.NS"].isna().all()
    assert pe_matrix["LIQUIDBEES.NS"].isna().all()
    
    # Assert Tier 1 indices (with own PE benchmarks) have non-NaN fundamental data
    assert not pe_matrix["^NSEI"].dropna().empty, "Tier 1 index should have PE data"
    assert not pe_matrix["^NSEBANK"].dropna().empty, "Tier 1 index should have PE data"
    
    # Assert Tier 3 ETFs (no exact-match benchmark) have NaN fundamentals
    # This prevents fabricated data from proxy mappings (e.g. PSUBNKBEES→^NSEBANK)
    if "JUNIORBEES.NS" in pe_matrix.columns:
        assert pe_matrix["JUNIORBEES.NS"].isna().all(), "Tier 3 ETF should have NaN PE"
    if "PSUBNKBEES.NS" in pe_matrix.columns:
        assert pe_matrix["PSUBNKBEES.NS"].isna().all(), "Tier 3 ETF should have NaN PE"

def test_signals_valuation_gate():
    """Verify that the signals engine de-escalates momentum signals under high valuation percentile."""
    from ui.views.signals import _evaluate_hysteresis_signal
    
    # Case A: Overvalued (pe_percentile = 0.95 >= 0.90) -> Downgraded
    signal_overvalued = _evaluate_hysteresis_signal(
        z_score=2.0,
        rank_pct=0.85,
        slope=0.01,
        regime="Trending Bull",
        prev_state="Neutral",
        bubble_z=1.0,
        pe_percentile=0.95
    )
    assert signal_overvalued["state"] == "Momentum"
    assert "Overvaluation Risk" in signal_overvalued["signal"]
    assert signal_overvalued["badge"] == "warning"
    assert "PE >= 90th percentile" in signal_overvalued["confidence"]
    
    # Case B: Fairly valued (pe_percentile = 0.50 < 0.90) -> Normal Momentum
    signal_fair = _evaluate_hysteresis_signal(
        z_score=2.0,
        rank_pct=0.85,
        slope=0.01,
        regime="Trending Bull",
        prev_state="Neutral",
        bubble_z=1.0,
        pe_percentile=0.50
    )
    assert signal_fair["state"] == "Momentum"
    assert "🟢 Momentum" in signal_fair["signal"]
    assert signal_fair["badge"] == "success"

def test_fetcher_price_bar_sanitization():
    """Verify that _sanitize_price_bars corrects open-high-low-close boundaries and zero/negative prices."""
    from data.fetcher import DataFetcher
    fetcher = DataFetcher()
    
    # Create a corrupted mock DataFrame
    df = pd.DataFrame({
        "Open": [10.0, -5.0, 10.0],
        "High": [8.0, 10.0, 10.0],  # row 0 has High < Open
        "Low": [12.0, 5.0, 8.0],   # row 0 has Low > Open
        "Close": [10.0, 10.0, 10.0],
        "Volume": [100.0, 100.0, -50.0]  # row 2 has negative volume
    })
    
    df_clean = fetcher._sanitize_price_bars(df, "MOCK_TICKER")
    
    # Assert negative Open was masked to NaN
    assert pd.isna(df_clean.loc[1, "Open"])
    
    # Assert High < Open was corrected to Open
    assert df_clean.loc[0, "High"] == 10.0
    
    # Assert Low > Open was corrected to Open/Close min
    assert df_clean.loc[0, "Low"] == 10.0
    
    # Assert negative Volume was corrected to 0
    assert df_clean.loc[2, "Volume"] == 0.0


def test_fetcher_lock_file():
    """Verify that fetch_all creates and cleans up the concurrency lock file."""
    from data.fetcher import DataFetcher
    from config.settings import CACHE_DIR
    from unittest.mock import patch
    
    lock_file = CACHE_DIR / "data.lock"
    if lock_file.exists():
        lock_file.unlink()
        
    fetcher = DataFetcher()
    # Mock universe to contain only one ticker to keep it fast
    fetcher.universe = {"MOCK_TICKER": {"class": "Broad Market - Benchmark", "type": "PR", "inception": "2000-01-01"}}
    
    with patch("yfinance.download", return_value=pd.DataFrame()) as mock_download:
        # Check that the lock file is created and cleaned up
        fetcher.fetch_all(end_date="2026-06-28")
        assert not lock_file.exists(), "Lock file should be deleted on completion"
        
        # Test locked scenario: create a fresh lock file and mock wait times
        lock_file.touch()
        with patch("time.sleep") as mock_sleep:
            res = fetcher.fetch_all(end_date="2026-06-28")
            # Should have attempted to sleep and wait
            assert mock_sleep.call_count > 0
            assert res == {}, "Should return empty dict on lock failure"
            
        # Cleanup
        if lock_file.exists():
            lock_file.unlink()


def test_fetcher_incremental_logic():
    """Verify that fetcher fetches incrementally if cached raw parquet exists."""
    from data.fetcher import DataFetcher
    from config.settings import CACHE_DIR
    from unittest.mock import patch
    
    fetcher = DataFetcher()
    # Mock universe
    fetcher.universe = {"MOCK_TICKER": {"class": "Broad Market - Benchmark", "type": "PR", "inception": "2000-01-01"}}
    
    # Create cached DataFrame
    dates = pd.date_range("2026-06-01", "2026-06-20", freq="D")
    df_cached = pd.DataFrame({
        "Open": [100.0] * len(dates),
        "High": [101.0] * len(dates),
        "Low": [99.0] * len(dates),
        "Close": [100.0] * len(dates),
        "Volume": [1000.0] * len(dates)
    }, index=dates)
    
    cache_file = CACHE_DIR / "MOCK_TICKER_raw.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df_cached.to_parquet(cache_file)
    
    try:
        # Scenario A: already up to date
        with patch("yfinance.download") as mock_download:
            res = fetcher.fetch_all(end_date="2026-06-20")
            assert mock_download.call_count == 0, "Should not call yfinance if already up to date"
            
        # Scenario B: needs incremental fetch
        # End date is 2026-06-25, so we expect start to be 2026-06-21
        with patch("yfinance.download", return_value=pd.DataFrame()) as mock_download:
            fetcher.fetch_all(end_date="2026-06-25")
            mock_download.assert_called_once()
            args, kwargs = mock_download.call_args
            assert kwargs["start"] == "2026-06-21"
            assert kwargs["end"] == "2026-06-25"
            
    finally:
        # Cleanup
        if cache_file.exists():
            cache_file.unlink()


