import pandas as pd
import numpy as np
from core.state_math import calculate_z_score, calculate_trend

class ArbitrationEngine:
    def __init__(self, price_series: pd.Series, rolling_returns: pd.Series = None):
        """
        Engine to generate actionable trading signals based on mathematical state.
        :param price_series: Master price series for the asset
        :param rolling_returns: Optional pre-calculated rolling returns. If none, we use price for z-score.
        """
        self.price_series = price_series.dropna()
        self.rolling_returns = rolling_returns.dropna() if rolling_returns is not None else self.price_series.pct_change(252).dropna()
        
        # Calculate state variables
        # We need at least 126 days (window/2) to have a valid Z-score with min_periods
        self.z_score_series = calculate_z_score(self.rolling_returns, window=252)
        self.trend_df = calculate_trend(self.price_series)
        
        # Check for data sufficiency
        self.has_sufficient_data = len(self.z_score_series.dropna()) > 0 and len(self.trend_df.dropna()) > 0

    def evaluate_current_state(self) -> dict:
        """
        Generates the 'Arbitration Hero Panel' metrics for the current day.
        """
        if not self.has_sufficient_data:
            return {
                "state": "⚪ INSUFFICIENT DATA",
                "veto_status": "Data window too short",
                "z_score": 0.0,
                "is_positive_trend": False
            }
            
        current_z = self.z_score_series.iloc[-1]
        current_trend = self.trend_df['is_positive_trend'].iloc[-1]
        
        # Base logic for Accumulation Zone
        # A deep discount (Z < -1.5) OR (Z < -1.0 and positive trend returning)
        if current_z < -1.5:
            state = "🟢 ACCUMULATION ZONE"
            veto = "Cleared by Deep Value"
        elif current_z < -1.0 and current_trend:
            state = "🟢 ACCUMULATION ZONE"
            veto = "Cleared by Value + Momentum"
        elif current_z > 1.5:
            state = "🔴 DISTRIBUTION ZONE"
            veto = "Overextended"
        else:
            state = "⚪ NEUTRAL HOLD"
            veto = "Standard Operating Environment"
            
        return {
            "state": state,
            "veto_status": veto,
            "z_score": current_z,
            "is_positive_trend": current_trend
        }

    def generate_action_grid(self, current_z: float, current_price: float, peak_price: float) -> dict:
        """
        Generates the flow multipliers and dry powder triggers.
        """
        if not self.has_sufficient_data:
            return {
                "sip": {"action": "WAIT", "multiplier": "1.0x", "rationale": "Insufficient history for signal generation."},
                "reserves": {"action": "HOLD", "drawdown": 0.0, "next_trigger": "N/A"}
            }
            
        drawdown = (current_price - peak_price) / peak_price if peak_price > 0 else 0
        
        # SIP Flow Logic
        if current_z < -1.5:
            sip_action = "SCALE UP"
            sip_multiplier = "1.5x - 2.0x"
            sip_rationale = f"Asset is trading at a significant {current_z:.2f} standard deviation discount."
        elif current_z > 1.5:
            sip_action = "HALT / REDUCE"
            sip_multiplier = "0.0x - 0.5x"
            sip_rationale = f"Asset is highly overextended ({current_z:.2f} SD). Avoid buying the top."
        else:
            sip_action = "NORMAL FLOW"
            sip_multiplier = "1.0x"
            sip_rationale = "Asset is within normal historical bounds. Continue baseline allocation."

        # Dry Powder Logic
        if drawdown < -0.20:
            reserve_action = "DEPLOY TRANCHE 1"
            next_trigger = f"{(drawdown - 0.05)*100:.1f}%"
        elif drawdown < -0.10:
            reserve_action = "PREPARE RESERVES"
            next_trigger = "-20.0%"
        else:
            reserve_action = "HOLD RESERVES"
            next_trigger = "-15.0%"

        return {
            "sip": {
                "action": sip_action,
                "multiplier": sip_multiplier,
                "rationale": sip_rationale
            },
            "reserves": {
                "action": reserve_action,
                "drawdown": drawdown,
                "next_trigger": next_trigger
            }
        }

    def run_inversion_matrix(self, threshold: float = -1.5) -> dict:
        """
        Calculates historical performance following similar Z-Score drops.
        Provides the 'Transparency Matrix' to build user confidence.
        """
        if not self.has_sufficient_data or self.z_score_series.empty:
            return {
                "events_found": 0,
                "false_positive_rate": 0.0,
                "history_table": pd.DataFrame(),
                "all_history_table": pd.DataFrame(),
                "veto_printout": "No historical events matching this severity found (Insufficient Data)."
            }
            
        # Find historical instances where Z-Score crossed below threshold
        below_thresh = self.z_score_series < threshold
        crosses = below_thresh & (~below_thresh.shift(1).fillna(False).infer_objects(copy=False))
        
        event_dates = self.z_score_series[crosses].index
        
        results = []
        for event_date in event_dates:
            # get_loc() on a tz-aware DatetimeIndex can return a slice or
            # boolean array instead of a plain int.  Normalise to int first.
            raw_loc = self.price_series.index.get_loc(event_date)
            if isinstance(raw_loc, slice):
                idx_loc = raw_loc.start
            elif isinstance(raw_loc, np.ndarray):
                # boolean mask — take first True position
                matches = np.where(raw_loc)[0]
                if len(matches) == 0:
                    continue
                idx_loc = int(matches[0])
            else:
                idx_loc = int(raw_loc)

            # Approximate days: 3M ~ 63 days, 6M ~ 126 days, 1Y ~ 252 days
            try:
                price_0 = float(self.price_series.iloc[idx_loc])

                # 3M outcome
                if idx_loc + 63 < len(self.price_series):
                    price_3m = float(self.price_series.iloc[idx_loc + 63])
                    end_date_3m = self.price_series.index[idx_loc + 63].strftime("%Y-%m-%d")
                    ret_3m = (price_3m / price_0) - 1
                else:
                    price_3m = np.nan
                    end_date_3m = "N/A"
                    ret_3m = np.nan

                # 6M outcome
                if idx_loc + 126 < len(self.price_series):
                    price_6m = float(self.price_series.iloc[idx_loc + 126])
                    end_date_6m = self.price_series.index[idx_loc + 126].strftime("%Y-%m-%d")
                    ret_6m = (price_6m / price_0) - 1
                else:
                    price_6m = np.nan
                    end_date_6m = "N/A"
                    ret_6m = np.nan

                # 1Y max drawdown
                if idx_loc + 252 < len(self.price_series):
                    future_1y_prices = self.price_series.iloc[idx_loc:idx_loc + 252]
                    max_dd_1y = float(
                        ((future_1y_prices - future_1y_prices.cummax()) / future_1y_prices.cummax()).min()
                    )
                else:
                    max_dd_1y = np.nan

                results.append({
                    "Start Date": event_date.strftime("%Y-%m-%d"),
                    "Start Price": round(price_0, 2),
                    "Z-Score": float(self.z_score_series.loc[event_date]),
                    "End Date (3M)": end_date_3m,
                    "End Price (3M)": round(price_3m, 2) if not np.isnan(price_3m) else np.nan,
                    "Next 3M Return": ret_3m,
                    "End Date (6M)": end_date_6m,
                    "End Price (6M)": round(price_6m, 2) if not np.isnan(price_6m) else np.nan,
                    "Next 6M Return": ret_6m,
                    "Max Drawdown (Next 1Y)": max_dd_1y
                })
            except Exception:
                continue


        df_results = pd.DataFrame(results)
        
        if df_results.empty:
            return {
                "events_found": 0,
                "false_positive_rate": 0.0,
                "history_table": pd.DataFrame(),
                "all_history_table": pd.DataFrame(),
                "veto_printout": "No historical events matching this severity found."
            }
            
        # Calculate false positive rate (defined as 6M return being negative)
        valid_6m = df_results['Next 6M Return'].dropna()
        if len(valid_6m) > 0:
            false_positives = (valid_6m < 0).sum()
            fpr = false_positives / len(valid_6m)
        else:
            fpr = 0.0
            
        printout = f"Historical False Positive Rate is {fpr*100:.1f}%. "
        if fpr < 0.2:
            printout += "Risk of immediate structural failure is low. Signal Approved."
        else:
            printout += "Elevated risk of continued drawdown. Scale in slowly."

        return {
            "events_found": len(df_results),
            "false_positive_rate": fpr,
            "history_table": df_results.tail(5).set_index("Start Date"),  # Show last 5 events
            "all_history_table": df_results.set_index("Start Date"),      # All events for full view
            "veto_printout": printout
        }

