import numpy as np
from core.features import calculate_cagr_nb, calculate_percentile_nb

def test_cagr():
    # Price doubles in 1 year (252 days)
    # CAGR = (2/1)^(252/252) - 1 = 1.0 (100%)
    cagr = calculate_cagr_nb(200.0, 100.0, 252)
    assert np.isclose(cagr, 1.0)

def test_percentile():
    history = np.array([10, 20, 30, 40, 50], dtype=np.float64)
    # 25 is greater than 10 and 20 (2 out of 5)
    # Percentile = (2/5) * 100 = 40.0
    p = calculate_percentile_nb(25.0, history)
    assert np.isclose(p, 40.0)

if __name__ == "__main__":
    test_cagr()
    test_percentile()
    print("All math tests passed!")
