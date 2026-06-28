import numpy as np
import vectorbt as vbt
from numba import njit
import math
from strategies.base import BaseStrategy

@njit(cache=True)
def rolling_returns_logic_nb(close, sma200, window, pct_window, entry_p, exit_p, max_value_trap_days):
    entries = np.zeros(close.shape, dtype=np.bool_)
    exits = np.zeros(close.shape, dtype=np.bool_)
    percentiles = np.full(close.shape, np.nan, dtype=np.float64)
    time_in_value = np.zeros(close.shape, dtype=np.int32)
    warmup_days = 252

    for col in range(close.shape[1]):
        cagr_buffer = np.empty(pct_window, dtype=np.float64)
        cagr_idx = 0
        in_pos = False
        consecutive_value_days = 0
        
        for t in range(window, close.shape[0]):
            price_t = close[t, col]
            price_t_w = close[t - window, col]
            if price_t_w <= 0:
                consecutive_value_days = 0
                time_in_value[t, col] = 0
                continue
                
            cagr = math.pow((price_t / price_t_w), (252.0 / window)) - 1.0
            cagr_buffer[cagr_idx % pct_window] = cagr
            cagr_idx += 1
            valid_cagr_count = cagr_idx if cagr_idx < pct_window else pct_window
            
            smaller_count = 0.0
            for i in range(valid_cagr_count):
                if cagr_buffer[i] < cagr:
                    smaller_count += 1.0
            
            current_percentile = (smaller_count / valid_cagr_count) * 100.0
            percentiles[t, col] = current_percentile
            
            if current_percentile < entry_p:
                consecutive_value_days += 1
            else:
                consecutive_value_days = 0
            time_in_value[t, col] = consecutive_value_days
            
            if valid_cagr_count >= warmup_days:
                if not in_pos:
                    curr_sma = sma200[t, col]
                    if not np.isnan(curr_sma) and price_t > curr_sma and current_percentile < entry_p:
                        entries[t, col] = True
                        in_pos = True
                else:
                    if current_percentile > exit_p or consecutive_value_days > max_value_trap_days:
                        exits[t, col] = True
                        in_pos = False
                        
    return entries, exits, percentiles, time_in_value

class RollingReturnsStrategy(BaseStrategy):
    @property
    def name(self):
        return "Rolling Returns"

    @property
    def description(self):
        return "Mean-reversion strategy evaluating deep value. Standardizes absolute CAGR into a historical percentile oscillator. Enters at < 15th percentile with 200-SMA filter."

    def get_default_params(self):
        return {
            "window": 756,
            "pct_window": 1260,
            "entry_p": 15.0,
            "exit_p": 85.0,
            "max_value_trap_days": 45
        }

    def run(self, data, params):
        """
        Run rolling returns mean-reversion strategy using VBT indicator factory.
        Expects data dict with 'close' (price DataFrame) and 'sma200' (200-day SMA DataFrame).
        """
        merged_params = {**self.get_default_params(), **params}
        close = data["close"] if isinstance(data, dict) else data
        sma200 = data.get("sma200", close.rolling(200).mean()) if isinstance(data, dict) else close.rolling(200).mean()
        
        result = RollingReturnsVBT.run(
            close, sma200,
            window=merged_params["window"],
            pct_window=merged_params["pct_window"],
            entry_p=merged_params["entry_p"],
            exit_p=merged_params["exit_p"],
            max_value_trap_days=merged_params["max_value_trap_days"]
        )
        return {
            "entries": result.entries,
            "exits": result.exits,
            "percentile": result.percentile,
            "time_in_value": result.time_in_value,
        }

# Legacy VectorBT Indicator for backward compatibility if needed
RollingReturnsVBT = vbt.IndicatorFactory(
    class_name="RollingReturns",
    short_name="rr",
    input_names=["close", "sma200"],
    param_names=["window", "pct_window", "entry_p", "exit_p", "max_value_trap_days"],
    output_names=["entries", "exits", "percentile", "time_in_value"]
).from_apply_func(
    rolling_returns_logic_nb,
    window=756,
    pct_window=1260,
    entry_p=15.0,
    exit_p=85.0,
    max_value_trap_days=45
)
