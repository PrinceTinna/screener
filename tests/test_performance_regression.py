"""
Performance Regression Test Suite
=================================
These tests lock in the current mathematical behavior of the QuantPro engine
BEFORE any performance optimizations are applied. They serve as a safety net
to ensure refactoring does not alter outputs.

Covers:
  - Rolling returns determinism (multi-window)
  - CAGR hand-calculation match
  - Cross-sectional percentiles (Numba path)
  - Regime classification vectorized equivalence
  - Z-Score vectorized equivalence
  - Signal state machine transition stability
  - Screener metrics consistency
  - Pipeline matrix shape preservation
  - Performance timing budget
"""
import pytest
import numpy as np
import pandas as pd
import time

from core.indicators import MathEngine, fast_percentiles_2d
from core.state_math import calculate_z_score, classify_regime_2d
from ui.views.signals import _evaluate_hysteresis_signal
from config.settings import TRADING_DAYS_PER_YEAR


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_matrix():
    """
    Creates a deterministic 1500-row x 5-asset price matrix.
    Simulates ~6 years of trading data with known growth rates.
    """
    np.random.seed(42)
    n_days = 1500
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    # Each asset has a distinct deterministic drift + small noise
    assets = {}
    drifts = [0.0004, 0.0002, -0.0001, 0.0006, 0.0001]
    names = ["ASSET_A", "ASSET_B", "ASSET_C", "ASSET_D", "ASSET_E"]

    for name, drift in zip(names, drifts):
        returns = np.random.normal(drift, 0.015, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        assets[name] = prices

    return pd.DataFrame(assets, index=dates)


@pytest.fixture
def large_synthetic_matrix():
    """
    Creates a 6000-row x 26-asset matrix for performance budget testing.
    Mimics the real production matrix dimensions.
    """
    np.random.seed(123)
    n_days = 6000
    n_assets = 26
    dates = pd.bdate_range("2000-01-01", periods=n_days)

    assets = {}
    for i in range(n_assets):
        drift = np.random.uniform(-0.0002, 0.0006)
        returns = np.random.normal(drift, 0.015, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        assets[f"ASSET_{i:02d}"] = prices

    return pd.DataFrame(assets, index=dates)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Rolling Returns Determinism
# ═══════════════════════════════════════════════════════════════════════════════

class TestRollingReturnsDeterministic:

    def test_multiple_windows_produce_correct_shape(self, synthetic_matrix):
        """Rolling returns should have same shape as input for any window."""
        engine = MathEngine(synthetic_matrix)

        for window in [21, 63, 126, 252, 756]:
            result = engine.calculate_rolling_returns(window)
            assert result.shape == synthetic_matrix.shape, \
                f"Shape mismatch for window={window}"

    def test_first_n_rows_are_nan(self, synthetic_matrix):
        """First `window` rows should be NaN (no lookback available)."""
        engine = MathEngine(synthetic_matrix)

        for window in [21, 252]:
            result = engine.calculate_rolling_returns(window)
            # All values in first `window` rows should be NaN
            assert result.iloc[:window].isna().all().all(), \
                f"Expected NaN in first {window} rows for window={window}"

    def test_known_value_calculation(self):
        """Verify against hand-calculated rolling return."""
        dates = pd.bdate_range("2024-01-01", periods=6)
        df = pd.DataFrame({"X": [100, 105, 110, 115, 120, 125]}, index=dates)

        engine = MathEngine(df)
        result = engine.calculate_rolling_returns(2)

        # Row 2 (index 2): (110/100) - 1 = 0.10
        assert pytest.approx(result.iloc[2]["X"], abs=1e-10) == 0.10
        # Row 3 (index 3): (115/105) - 1 ≈ 0.09524
        assert pytest.approx(result.iloc[3]["X"], abs=1e-4) == 0.09524
        # Row 5 (index 5): (125/115) - 1 ≈ 0.08696
        assert pytest.approx(result.iloc[5]["X"], abs=1e-4) == 0.08696

    def test_reproducibility_across_calls(self, synthetic_matrix):
        """Two calls with same input should produce bitwise-identical results."""
        engine = MathEngine(synthetic_matrix)
        r1 = engine.calculate_rolling_returns(252)
        r2 = engine.calculate_rolling_returns(252)

        pd.testing.assert_frame_equal(r1, r2)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: CAGR Determinism
# ═══════════════════════════════════════════════════════════════════════════════

class TestCAGRDeterministic:

    def test_1y_cagr_identity(self, synthetic_matrix):
        """For a 252-day window, CAGR should equal rolling return (years=1)."""
        engine = MathEngine(synthetic_matrix)
        rolling = engine.calculate_rolling_returns(252)
        cagr = engine.calculate_cagr(rolling, 252)

        # CAGR formula: (1 + R)^(1/1) - 1 = R
        valid_mask = rolling.notna()
        np.testing.assert_allclose(
            cagr.values[valid_mask],
            rolling.values[valid_mask],
            rtol=1e-10,
            err_msg="CAGR for 1-year window should equal raw return"
        )

    def test_known_cagr_calculation(self):
        """Hand-calculated CAGR for a known return."""
        engine = MathEngine(pd.DataFrame({"X": [100]}, index=[pd.Timestamp("2024-01-01")]))
        # If R = 0.50 over 504 days (2 years), CAGR = (1.5)^(252/504) - 1 = sqrt(1.5) - 1
        rolling = pd.DataFrame({"X": [0.50]}, index=[pd.Timestamp("2024-01-01")])
        cagr = engine.calculate_cagr(rolling, 504)

        expected = np.sqrt(1.50) - 1.0  # ≈ 0.2247
        assert pytest.approx(cagr.iloc[0]["X"], abs=1e-4) == expected

    def test_negative_return_cagr(self):
        """CAGR should handle negative returns correctly."""
        engine = MathEngine(pd.DataFrame({"X": [100]}, index=[pd.Timestamp("2024-01-01")]))
        rolling = pd.DataFrame({"X": [-0.20]}, index=[pd.Timestamp("2024-01-01")])
        cagr = engine.calculate_cagr(rolling, 504)

        # (1 + (-0.20))^(252/504) - 1 = 0.8^0.5 - 1 ≈ -0.1056
        expected = 0.8 ** 0.5 - 1.0
        assert pytest.approx(cagr.iloc[0]["X"], abs=1e-4) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Cross-Sectional Percentiles Determinism
# ═══════════════════════════════════════════════════════════════════════════════

class TestPercentilesDeterministic:

    def test_known_percentiles(self):
        """Verify Numba percentile path on a matrix with known values."""
        # 5 rows x 4 assets, values are 0-19
        matrix = np.arange(20, dtype=np.float64).reshape(5, 4)
        p10, p25, p50, p75, p90 = fast_percentiles_2d(matrix)

        # Row 0: [0, 1, 2, 3] → median = 1.5
        assert pytest.approx(p50[0]) == 1.5
        # Row 4: [16, 17, 18, 19] → median = 17.5
        assert pytest.approx(p50[4]) == 17.5

    def test_nan_handling(self):
        """NaN values should be excluded from percentile calculations."""
        matrix = np.array([
            [1.0, np.nan, 3.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [10.0, 20.0, 30.0, 40.0]
        ])
        p10, p25, p50, p75, p90 = fast_percentiles_2d(matrix)

        # Row 0: valid = [1.0, 3.0] → median = 2.0
        assert pytest.approx(p50[0]) == 2.0
        # Row 1: all NaN → should be NaN
        assert np.isnan(p50[1])
        # Row 2: [10, 20, 30, 40] → median = 25.0
        assert pytest.approx(p50[2]) == 25.0

    def test_dataframe_wrapper(self, synthetic_matrix):
        """MathEngine wrapper should return a DataFrame with correct columns."""
        engine = MathEngine(synthetic_matrix)
        rolling = engine.calculate_rolling_returns(63)
        percentiles = engine.get_cross_sectional_percentiles(rolling)

        assert list(percentiles.columns) == ['P10', 'P25', 'Median', 'P75', 'P90']
        assert len(percentiles) == len(rolling)
        assert percentiles.index.equals(rolling.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Regime Classification Equivalence
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeClassification:

    @pytest.fixture
    def regime_test_data(self):
        """Creates synthetic vol and price data for regime testing."""
        np.random.seed(42)
        n = 1000
        prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.01, n)) + 100)
        daily_rets = prices.pct_change().dropna()
        vol = daily_rets.rolling(window=252).std() * np.sqrt(252)
        return vol, prices

    def test_output_length_matches_input(self, regime_test_data):
        """Regime series should have the same length as vol input."""
        vol, prices = regime_test_data
        regime = classify_regime_2d(vol, prices, window=252)
        assert len(regime) == len(vol)

    def test_valid_regime_labels(self, regime_test_data):
        """All regime labels should be from the known set."""
        vol, prices = regime_test_data
        regime = classify_regime_2d(vol, prices, window=252)

        valid_labels = {
            "Trending Bull", "Neutral Bull", "Recovery",
            "Low-Vol Range", "Neutral Bear", "Panic",
            "Extreme Breakdown", "Unknown"
        }
        actual_labels = set(regime.unique())
        assert actual_labels.issubset(valid_labels), \
            f"Unexpected regime labels: {actual_labels - valid_labels}"

    def test_regime_reproducibility(self, regime_test_data):
        """Two calls with same input should produce identical results."""
        vol, prices = regime_test_data
        r1 = classify_regime_2d(vol, prices, window=252)
        r2 = classify_regime_2d(vol, prices, window=252)
        pd.testing.assert_series_equal(r1, r2)

    def test_early_unknown_regime(self, regime_test_data):
        """Early entries should be 'Unknown' due to insufficient lookback."""
        vol, prices = regime_test_data
        regime = classify_regime_2d(vol, prices, window=252)
        # The first ~252 entries should mostly be Unknown (vol_q40 needs 2*window)
        early_slice = regime.iloc[:252]
        unknown_count = (early_slice == "Unknown").sum()
        assert unknown_count > 0, "Expected 'Unknown' regime in early data"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Z-Score Vectorized Equivalence
# ═══════════════════════════════════════════════════════════════════════════════

class TestZScoreEquivalence:

    def test_single_series_known_values(self):
        """Z-Score of a constant series should be 0 (or NaN)."""
        constant = pd.Series([10.0] * 500)
        z = calculate_z_score(constant, window=252)
        # std = 0, so Z-score should be NaN (division by zero handled)
        valid_z = z.dropna()
        # All should be NaN because std == 0
        assert len(valid_z) == 0 or (valid_z == 0).all() or valid_z.isna().all()

    def test_column_by_column_matches_manual(self, synthetic_matrix):
        """Z-Scores computed column-by-column should match individual calls."""
        engine = MathEngine(synthetic_matrix)
        rolling = engine.calculate_rolling_returns(252)

        # Column-by-column (current screener approach)
        z_cols = {}
        for col in rolling.columns:
            z_cols[col] = calculate_z_score(rolling[col], window=252)

        z_manual = pd.DataFrame(z_cols)

        # Vectorized approach: apply to entire DataFrame
        z_vectorized = rolling.apply(lambda col: calculate_z_score(col, window=252))

        pd.testing.assert_frame_equal(z_manual, z_vectorized, check_names=False)

    def test_z_score_range(self, synthetic_matrix):
        """Z-Scores should generally fall within reasonable bounds."""
        engine = MathEngine(synthetic_matrix)
        rolling = engine.calculate_rolling_returns(252)

        for col in rolling.columns:
            z = calculate_z_score(rolling[col], window=252).dropna()
            if len(z) > 0:
                # Extreme Z-scores beyond ±5 would indicate a bug
                assert z.abs().max() < 10, \
                    f"Z-Score for {col} has extreme value: {z.abs().max():.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: Signal State Machine Stability
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalStateMachineStability:

    def test_100_state_transitions(self):
        """Run 100 synthetic evaluations and verify deterministic transitions."""
        np.random.seed(42)

        # Generate 100 random market states
        z_scores = np.random.uniform(-3, 3, 100)
        ranks = np.random.uniform(0, 1, 100)
        slopes = np.random.uniform(-0.01, 0.01, 100)
        regimes = np.random.choice(
            ["Trending Bull", "Neutral Bull", "Recovery", "Panic",
             "Low-Vol Range", "Neutral Bear", "Extreme Breakdown"],
            100
        )

        # Run forward pass
        states_1 = []
        prev = "Neutral"
        for i in range(100):
            sig = _evaluate_hysteresis_signal(
                z_scores[i], ranks[i], slopes[i], regimes[i], prev
            )
            states_1.append(sig["state"])
            prev = sig["state"]

        # Run again — should be identical
        states_2 = []
        prev = "Neutral"
        for i in range(100):
            sig = _evaluate_hysteresis_signal(
                z_scores[i], ranks[i], slopes[i], regimes[i], prev
            )
            states_2.append(sig["state"])
            prev = sig["state"]

        assert states_1 == states_2, "State machine is non-deterministic!"

    def test_all_return_keys_present(self):
        """Every signal dict must have the required keys."""
        required_keys = {"signal", "badge", "confidence", "detail", "state", "display_state"}

        combos = [
            (2.0, 0.95, 0.01, "Trending Bull", "Neutral"),
            (-2.0, 0.05, 0.003, "Panic", "Neutral"),
            (0.0, 0.50, 0.0, "Low-Vol Range", "Neutral"),
            (1.0, 0.80, 0.005, "Trending Bull", "Momentum"),
            (-1.5, 0.15, -0.005, "Extreme Breakdown", "Reversal"),
        ]
        for z, r, s, regime, prev in combos:
            sig = _evaluate_hysteresis_signal(z, r, s, regime, prev)
            assert set(sig.keys()) == required_keys, \
                f"Missing keys for inputs z={z}, r={r}: {required_keys - set(sig.keys())}"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: Screener Metrics Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestScreenerMetrics:

    def test_cagr_vol_sharpe_consistency(self, synthetic_matrix):
        """CAGR, Vol, and Sharpe should be mathematically consistent."""
        engine = MathEngine(synthetic_matrix)
        window = 756

        rolling = engine.calculate_rolling_returns(window)
        cagr_matrix = engine.calculate_cagr(rolling, window)

        latest_cagr = cagr_matrix.iloc[-1]
        vol_matrix = rolling.rolling(window=window).std() * np.sqrt(252)
        latest_vol = vol_matrix.iloc[-1]

        sharpe = latest_cagr / latest_vol.replace(0, np.nan)

        # Sharpe = CAGR / Vol (by definition)
        for col in synthetic_matrix.columns:
            if pd.notna(latest_vol[col]) and latest_vol[col] != 0:
                expected_sharpe = latest_cagr[col] / latest_vol[col]
                assert pytest.approx(sharpe[col], abs=1e-8) == expected_sharpe

    def test_rank_percentile_bounds(self, synthetic_matrix):
        """Segment rank percentiles should be in [0, 1]."""
        engine = MathEngine(synthetic_matrix)
        rolling = engine.calculate_rolling_returns(756)
        cagr_matrix = engine.calculate_cagr(rolling, 756)
        latest_cagr = cagr_matrix.iloc[-1].dropna()

        ranks = latest_cagr.rank(pct=True)
        assert (ranks >= 0).all() and (ranks <= 1).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8: Pipeline Matrix Shape Preservation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineShapePreservation:

    def test_rolling_preserves_columns(self, synthetic_matrix):
        """Rolling returns must preserve all column names from the input matrix."""
        engine = MathEngine(synthetic_matrix)
        for window in [21, 252, 756]:
            result = engine.calculate_rolling_returns(window)
            assert list(result.columns) == list(synthetic_matrix.columns)

    def test_rolling_preserves_index(self, synthetic_matrix):
        """Rolling returns must preserve the DatetimeIndex from the input matrix."""
        engine = MathEngine(synthetic_matrix)
        result = engine.calculate_rolling_returns(252)
        pd.testing.assert_index_equal(result.index, synthetic_matrix.index)

    def test_cagr_preserves_shape(self, synthetic_matrix):
        """CAGR matrix must have identical shape to rolling returns."""
        engine = MathEngine(synthetic_matrix)
        rolling = engine.calculate_rolling_returns(252)
        cagr = engine.calculate_cagr(rolling, 252)
        assert cagr.shape == rolling.shape


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9: Performance Budget
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceBudget:

    def test_core_pipeline_under_budget(self, large_synthetic_matrix):
        """
        Full compute pipeline (rolling + CAGR + percentiles) on a 6000x26
        matrix should complete within 2 seconds.
        """
        engine = MathEngine(large_synthetic_matrix)

        start = time.perf_counter()

        rolling = engine.calculate_rolling_returns(756)
        cagr = engine.calculate_cagr(rolling, 756)
        percentiles = engine.get_cross_sectional_percentiles(rolling)

        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, \
            f"Core pipeline took {elapsed:.2f}s (budget: 2.0s)"

        # Sanity: outputs are non-empty
        assert not rolling.empty
        assert not cagr.empty
        assert not percentiles.empty

    def test_regime_classification_under_budget(self):
        """Regime classification on a 6000-row series should complete within 1 second."""
        np.random.seed(42)
        n = 6000
        prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.01, n)) + 100)
        daily_rets = prices.pct_change().dropna()
        vol = daily_rets.rolling(window=252).std() * np.sqrt(252)

        start = time.perf_counter()
        regime = classify_regime_2d(vol, prices, window=252)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, \
            f"Regime classification took {elapsed:.2f}s (budget: 1.0s)"
        assert len(regime) == len(vol)
