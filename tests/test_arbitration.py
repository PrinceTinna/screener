import pytest
import pandas as pd
import numpy as np
from strategies.arbitration import ArbitrationEngine

@pytest.fixture
def sample_price_data():
    """Generates synthetic price data with a clear flash crash at the end."""
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    
    # Create a stable uptrend
    prices = np.linspace(100, 200, 1000)
    
    # Induce a flash crash in the last 10 days
    prices[-10:] = prices[-10:] * 0.70 
    
    return pd.Series(prices, index=dates)

def test_accumulation_signal_on_crash(sample_price_data):
    """
    Asserts the Z-Score accurately triggers an "Accumulation" state 
    and that the multiplier logic scales correctly during a flash crash.
    """
    engine = ArbitrationEngine(sample_price_data)
    
    # 1. Evaluate State
    state = engine.evaluate_current_state()
    assert "ACCUMULATION ZONE" in state['state']
    assert state['z_score'] < -1.5
    
    # 2. Evaluate Multipliers
    current_price = sample_price_data.iloc[-1]
    peak_price = sample_price_data.max()
    
    action_grid = engine.generate_action_grid(state['z_score'], current_price, peak_price)
    
    assert action_grid['sip']['action'] == "SCALE UP"
    assert "1.5x" in action_grid['sip']['multiplier']
    
    assert action_grid['reserves']['action'] == "DEPLOY TRANCHE 1"
    assert action_grid['reserves']['drawdown'] <= -0.20

def test_distribution_signal_on_meltup():
    """Test the inverse logic on an overextended asset."""
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    prices = np.linspace(100, 120, 1000)
    prices[-10:] = prices[-10:] * 1.50 # Melt up
    
    meltup_data = pd.Series(prices, index=dates)
    engine = ArbitrationEngine(meltup_data)
    
    state = engine.evaluate_current_state()
    assert "DISTRIBUTION ZONE" in state['state']
    assert state['z_score'] > 1.5
    
    action_grid = engine.generate_action_grid(state['z_score'], meltup_data.iloc[-1], meltup_data.max())
    assert action_grid['sip']['action'] == "HALT / REDUCE"

def test_inversion_matrix_false_positives(sample_price_data):
    """Test that the inversion matrix correctly identifies historical events."""
    engine = ArbitrationEngine(sample_price_data)
    matrix = engine.run_inversion_matrix(threshold=-1.0)
    
    assert 'events_found' in matrix
    assert 'false_positive_rate' in matrix
    assert 'history_table' in matrix
