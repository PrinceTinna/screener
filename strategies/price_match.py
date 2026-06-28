import numpy as np
import vectorbt as vbt
from numba import njit
from strategies.base import BaseStrategy

@njit(cache=True)
def price_match_njit(close, lookback, rolling_peak, tolerance):
    entries = np.zeros(close.shape, dtype=np.bool_)
    for col in range(close.shape[1]):
        for t in range(lookback, close.shape[0]):
            price_t = close[t, col]
            if rolling_peak:
                target_price = np.max(close[t-lookback:t, col])
            else:
                target_price = close[t-lookback, col]
            if target_price == 0:
                continue
            ceiling = target_price * (1.0 + tolerance / 100.0)
            if price_t <= ceiling:
                entries[t, col] = True
    return entries

class PriceMatchStrategy(BaseStrategy):
    @property
    def name(self):
        return "Price Match Reversion"

    @property
    def description(self):
        return "Matches current price against a historical level or rolling peak. Signals identify assets trading significantly below their recent reference points, betting on a snap-back."

    def get_default_params(self):
        return {
            "lookback": 126,
            "rolling_peak": False,
            "tolerance": 0.3
        }

    def run(self, data, params):
        """Run price match strategy using VBT indicator factory."""
        merged_params = {**self.get_default_params(), **params}
        result = PriceMatchVBT.run(
            data,
            lookback=merged_params["lookback"],
            rolling_peak=merged_params["rolling_peak"],
            tolerance=merged_params["tolerance"]
        )
        return {"entries": result.entries}

PriceMatchVBT = vbt.IndicatorFactory(
    class_name="PriceMatch",
    short_name="pm",
    input_names=["close"],
    param_names=["lookback", "rolling_peak", "tolerance"],
    output_names=["entries"]
).from_apply_func(
    price_match_njit,
    lookback=14,
    rolling_peak=False,
    tolerance=0.0
)
