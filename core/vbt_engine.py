import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit

@njit(cache=True)
def generate_fixed_exits_nb(entries, hold_period):
    """
    Numba-optimized fixed-hold exit generator.
    For each True in entries, sets True in exits exactly 'hold_period' days later.
    """
    exits = np.zeros(entries.shape, dtype=np.bool_)
    for col in range(entries.shape[1]):
        for t in range(entries.shape[0]):
            if entries[t, col]:
                exit_t = t + hold_period
                if exit_t < entries.shape[0]:
                    exits[exit_t, col] = True
    return exits

def run_backtest(close, entries, hold_period=None, exits=None, open_price=None,
                 init_cash=100000, fees=0.001, slippage=0.001,
                 lookback_buffer=None, sizing_mode='Fixed Capital', short_entries=None,
                 short_exits=None):
    """
    Core Portfolio Engine for VectorBT.
    Supports custom exit signals and Next-Day-Open execution.
    """
    if entries.sum().sum() == 0 and (short_entries is None or short_entries.sum().sum() == 0):
        return None

    # 1. Exit Generation Logic
    if exits is None and hold_period is not None:
        exits_val = generate_fixed_exits_nb(entries.values, int(hold_period))
        exits = pd.DataFrame(exits_val, index=entries.index, columns=entries.columns)
    
    if short_entries is not None and short_exits is None and hold_period is not None:
        short_exits_val = generate_fixed_exits_nb(short_entries.values, int(hold_period))
        short_exits = pd.DataFrame(short_exits_val, index=short_entries.index, columns=short_entries.columns)

    # 2. Lookback Buffering (Slicing) & Look-Ahead Bias Prevention (Shift)
    def _slice(df):
        return df.iloc[lookback_buffer:] if (lookback_buffer and lookback_buffer > 0) else df

    def _shift_signals(df):
        if df is None: return None
        return pd.DataFrame(df).shift(1).fillna(False)

    close_final    = _slice(close)
    entries_final  = _slice(_shift_signals(entries))
    exits_final    = _slice(_shift_signals(exits))
    
    exec_price = _slice(open_price) if open_price is not None else close_final

    # 3. Portfolio Simulation
    size      = np.inf if sizing_mode == 'Fixed Capital per Trade' else 1
    size_type = 'Value' if sizing_mode == 'Fixed Capital per Trade' else 'Amount'

    vbt_kwargs = {
        "close": close_final,
        "entries": entries_final,
        "exits": exits_final,
        "init_cash": init_cash,
        "fees": fees,
        "slippage": slippage,
        "freq": '1D',
        "size": size,
        "size_type": size_type,
        "accumulate": False
    }
    
    if open_price is not None:
        vbt_kwargs["price"] = exec_price

    if short_entries is not None:
        vbt_kwargs["short_entries"] = _slice(_shift_signals(short_entries))
        vbt_kwargs["short_exits"] = _slice(_shift_signals(short_exits))
        pf = vbt.Portfolio.from_signals(**vbt_kwargs)
    else:
        vbt_kwargs["direction"] = 0
        pf = vbt.Portfolio.from_signals(**vbt_kwargs)

    return pf

def run_pairs_backtest(close_a, close_b, sig_df, beta, open_a=None, open_b=None, 
                       init_cash=100000, fees=0.001, slippage=0.001):
    """
    Pairs trading execution simulator.
    """
    close_df = pd.concat([close_a, close_b], axis=1, keys=['A', 'B'])
    
    if open_a is not None and open_b is not None:
        exec_price = pd.concat([open_a, open_b], axis=1, keys=['A', 'B'])
        trade_signals = sig_df.shift(1).fillna(False)
    else:
        exec_price = close_df
        trade_signals = sig_df.shift(1).fillna(False)

    long_entries = trade_signals['long_entries']
    long_exits = trade_signals['long_exits']
    short_entries = trade_signals['short_entries']
    short_exits = trade_signals['short_exits']
    
    entries_df = pd.concat([short_entries, long_entries], axis=1, keys=['A', 'B'])
    exits_df = pd.concat([short_exits, long_exits], axis=1, keys=['A', 'B'])
    short_entries_df = pd.concat([long_entries, short_entries], axis=1, keys=['A', 'B'])
    short_exits_df = pd.concat([long_exits, short_exits], axis=1, keys=['A', 'B'])
    
    val_B = init_cash / (1 + beta)
    val_A = beta * val_B
    
    size_df = pd.concat([pd.Series(val_A, index=close_a.index), pd.Series(val_B, index=close_b.index)], axis=1, keys=['A', 'B'])
    
    pf = vbt.Portfolio.from_signals(
        close=close_df,
        price=exec_price,
        entries=entries_df,
        exits=exits_df,
        short_entries=short_entries_df,
        short_exits=short_exits_df,
        size=size_df,
        size_type='Value',
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq='1D',
        accumulate=False
    )
    
    return pf
