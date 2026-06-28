import pytest
import pandas as pd
import numpy as np
from core.validators import validate_matrix_shape, validate_inception_alignment

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
    
    # Assert Equity Index and ETF mapping consistency
    # JUNIORBEES.NS is generated based on ^NSEI config in seed, they should both have non-NaN values
    assert not pe_matrix["^NSEI"].dropna().empty
    assert not pe_matrix["JUNIORBEES.NS"].dropna().empty

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

