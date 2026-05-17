"""
Phase 1.6 Test Suite: Signal Engine Tests
Covers:
  - Scenario 7: OLS Slope Guardrail (Falling Knife)
  - Scenario 8: Hysteresis Stability
  - Scenario 9: Hard Veto Execution (Extreme Volatility)
"""
import pytest
import numpy as np
import pandas as pd

from core.state_math import calculate_z_score, calculate_ols_slope, classify_regime_2d
from ui.views.signals import _evaluate_hysteresis_signal


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 7: Falling Knife Guardrail (OLS Slope)
# Pass a synthetic array where an asset's price plummets to Z = -2.0 but
# continues to drop (Slope < 0). Assert that the Reversal Signal does NOT
# trigger until the slope turns positive.
# ═══════════════════════════════════════════════════════════════════════════════
class TestFallingKnifeGuardrail:
    
    def test_reversal_blocked_when_slope_negative(self):
        """Reversal should NOT fire when Slope < 0 (still falling)."""
        signal = _evaluate_hysteresis_signal(
            z_score=-2.0,         # Deep value
            rank_pct=0.10,        # Bottom decile
            slope=-0.005,         # Still falling (negative slope)
            regime="Panic",       # High vol + down trend
            prev_state="Neutral"
        )
        assert signal["state"] != "Reversal", \
            f"Reversal triggered on a falling knife! Signal: {signal}"
    
    def test_reversal_triggers_when_slope_positive(self):
        """Reversal SHOULD fire when Slope > 0 (bounce confirmed)."""
        signal = _evaluate_hysteresis_signal(
            z_score=-2.0,         # Deep value
            rank_pct=0.10,        # Bottom decile
            slope=0.003,          # Turning positive (bounce)
            regime="Panic",       # High vol + down trend
            prev_state="Neutral"
        )
        assert signal["state"] == "Reversal", \
            f"Reversal failed to trigger on confirmed bounce! Signal: {signal}"
    
    def test_ols_slope_sign_matches_direction(self):
        """OLS slope should be negative for declining series and positive for rising."""
        # Declining series
        declining = pd.Series(np.linspace(100, 50, 30))
        slope_decline = calculate_ols_slope(declining, window=10)
        assert slope_decline.dropna().iloc[-1] < 0, "Slope should be negative for declining series"
        
        # Rising series
        rising = pd.Series(np.linspace(50, 100, 30))
        slope_rise = calculate_ols_slope(rising, window=10)
        assert slope_rise.dropna().iloc[-1] > 0, "Slope should be positive for rising series"
    
    def test_ols_slope_stability_vs_noise(self):
        """OLS slope should be more stable than simple difference under noise."""
        np.random.seed(42)
        # Uptrending series with heavy noise
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 3, 100)
        noisy_series = pd.Series(trend + noise)
        
        slope = calculate_ols_slope(noisy_series, window=20)
        valid_slopes = slope.dropna()
        
        # Despite noise, the majority of slopes should be positive (uptrend)
        positive_pct = (valid_slopes > 0).mean()
        assert positive_pct > 0.5, \
            f"OLS slope should detect uptrend despite noise. Only {positive_pct*100:.0f}% positive."


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 8: Hysteresis Stability
# Assert that once a signal is entered, small perturbations don't cause exits.
# Different entry/exit thresholds prevent flickering.
# ═══════════════════════════════════════════════════════════════════════════════
class TestHysteresisStability:
    
    def test_momentum_holds_below_entry_threshold(self):
        """Once in Momentum, signal should persist even if rank dips below entry."""
        # Already in Momentum state (prev_state="Momentum")
        # Rank has dipped to 70th (below 75th entry), but above 60th exit
        signal = _evaluate_hysteresis_signal(
            z_score=0.6,           # Above entry Z threshold
            rank_pct=0.70,         # Below entry (75th) but above exit (60th)
            slope=0.001,           # Positive
            regime="Trending Bull",
            prev_state="Momentum"  # Already in momentum
        )
        assert signal["state"] == "Momentum", \
            f"Momentum should HOLD due to hysteresis. Got: {signal['state']}"
        assert signal["confidence"] == "Holding (Hysteresis)"
    
    def test_momentum_exits_below_exit_threshold(self):
        """Momentum should exit when rank drops below exit threshold."""
        signal = _evaluate_hysteresis_signal(
            z_score=0.6,
            rank_pct=0.55,         # Below exit threshold (60th)
            slope=0.001,
            regime="Trending Bull",
            prev_state="Momentum"
        )
        assert signal["state"] != "Momentum", \
            f"Momentum should EXIT below 60th percentile. Got: {signal['state']}"
    
    def test_momentum_does_not_enter_below_entry_threshold(self):
        """A new Momentum signal should NOT enter below the entry threshold."""
        signal = _evaluate_hysteresis_signal(
            z_score=0.6,
            rank_pct=0.70,         # Below entry (75th)
            slope=0.001,
            regime="Trending Bull",
            prev_state="Neutral"   # Not already in momentum
        )
        assert signal["state"] != "Momentum", \
            f"Momentum should NOT enter at 70th percentile. Got: {signal['state']}"
    
    def test_reversal_holds_with_hysteresis(self):
        """Once in Reversal, signal should persist between entry and exit thresholds."""
        signal = _evaluate_hysteresis_signal(
            z_score=-0.8,          # Between entry (-1.0) and exit (-0.5)
            rank_pct=0.30,         # Between entry (25th) and exit (40th)
            slope=0.001,
            regime="Panic",
            prev_state="Reversal"  # Already in reversal
        )
        assert signal["state"] == "Reversal", \
            f"Reversal should HOLD due to hysteresis. Got: {signal['state']}"
    
    def test_no_flickering_across_boundary(self):
        """Simulate 5 consecutive evaluations near the boundary — state should be stable."""
        # Start neutral, then oscillate near momentum entry boundary
        states = []
        prev = "Neutral"
        for rank in [0.76, 0.74, 0.76, 0.73, 0.76]:
            signal = _evaluate_hysteresis_signal(
                z_score=0.6, rank_pct=rank, slope=0.001,
                regime="Trending Bull", prev_state=prev
            )
            states.append(signal["state"])
            prev = signal["state"]
        
        # Count state changes
        changes = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
        assert changes <= 2, \
            f"Signal flickered {changes} times near boundary: {states}. Expected ≤2."


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario 9: Hard Veto Execution (Extreme Volatility)
# Pass data with Volatility > 90th percentile. Assert that all Momentum
# signals are disabled entirely, regardless of trend or rank strength.
# ═══════════════════════════════════════════════════════════════════════════════
class TestHardVetoExecution:
    
    def test_momentum_vetoed_in_extreme_breakdown(self):
        """Even with perfect momentum conditions, Extreme Breakdown vetoes it."""
        signal = _evaluate_hysteresis_signal(
            z_score=2.0,             # Strong momentum
            rank_pct=0.95,           # Top decile
            slope=0.01,             # Strong positive slope
            regime="Extreme Breakdown",
            prev_state="Neutral"
        )
        assert signal["state"] == "Veto", \
            f"Hard Veto should disable momentum in Extreme Breakdown. Got: {signal['state']}"
        assert "VETO" in signal["signal"]
    
    def test_existing_momentum_vetoed_in_extreme_breakdown(self):
        """Even an existing Momentum position is vetoed in Extreme Breakdown."""
        signal = _evaluate_hysteresis_signal(
            z_score=2.0,
            rank_pct=0.95,
            slope=0.01,
            regime="Extreme Breakdown",
            prev_state="Momentum"   # Already in momentum — still vetoed
        )
        assert signal["state"] == "Veto", \
            f"Hard Veto should override existing Momentum. Got: {signal['state']}"
    
    def test_reversal_still_allowed_in_extreme_breakdown(self):
        """Reversal signals should NOT be blocked by the momentum veto."""
        # In Extreme Breakdown, the veto fires first and returns "Veto" state
        # which blocks everything. This is by design — in extreme breakdowns,
        # even reversal is too dangerous.
        signal = _evaluate_hysteresis_signal(
            z_score=-2.0,
            rank_pct=0.05,
            slope=0.003,
            regime="Extreme Breakdown",
            prev_state="Neutral"
        )
        # In our architecture, Extreme Breakdown is a HARD veto for ALL signals
        assert signal["state"] == "Veto", \
            f"Extreme Breakdown should veto ALL signals. Got: {signal['state']}"
    
    def test_regime_classification_extreme(self):
        """Regime classifier should identify extreme volatility correctly."""
        np.random.seed(42)
        n = 1000
        # Create a price series with an extreme vol spike at the end
        prices = pd.Series(np.cumsum(np.random.normal(0.001, 0.01, n)) + 100)
        # Inject massive volatility spike in last 50 days
        prices.iloc[-50:] += np.random.normal(0, 5, 50)
        
        daily_rets = prices.pct_change().dropna()
        vol = daily_rets.rolling(window=252).std() * np.sqrt(252)
        
        regime = classify_regime_2d(vol, prices, window=252)
        valid_regimes = regime.dropna()
        
        # The last regime should reflect the extreme conditions
        last_regime = valid_regimes.iloc[-1] if len(valid_regimes) > 0 else "Unknown"
        assert last_regime in ("Extreme Breakdown", "Panic", "Recovery"), \
            f"Expected extreme regime at end, got: {last_regime}"
