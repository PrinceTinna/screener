import pytest
import numpy as np
import pandas as pd
from core.indicators import fast_percentiles_2d, MathEngine

def test_fast_percentiles_2d():
    """Scenario 1: N-Dimensional Broadcasting Test"""
    # Create simple 2D Array: 3 assets, 4 days
    matrix = np.array([
        [0.01, 0.02, 0.03],
        [0.05, np.nan, 0.05],
        [-0.01, -0.02, 0.0],
        [0.10, 0.20, 0.30]
    ])
    
    p10, p25, p50, p75, p90 = fast_percentiles_2d(matrix)
    
    # Assert output shapes remain unchanged (1D vector of length 4)
    assert p50.shape == (4,)
    
    # Check median logic ignoring NaNs
    assert p50[1] == 0.05
    assert p50[0] == 0.02

def test_calculate_rolling_returns():
    """Basic math engine validation"""
    dates = pd.date_range("2020-01-01", periods=10)
    # Price doubles evenly
    df = pd.DataFrame({'AssetA': [100, 110, 120, 130, 200]}, index=dates[:5])
    
    engine = MathEngine(df)
    # 2 day rolling return
    ret = engine.calculate_rolling_returns(2)
    
    assert ret.shape == (5, 1)
    # Row 2 (index 2): (120/100) - 1.0 = 0.2
    assert pytest.approx(ret.iloc[2]['AssetA']) == 0.2
