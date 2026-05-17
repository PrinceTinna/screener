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
                "veto_printout": "No historical events matching this severity found (Insufficient Data)."
            }
            
        # Find historical instances where Z-Score crossed below threshold
        below_thresh = self.z_score_series < threshold
        crosses = below_thresh & (~below_thresh.shift(1).fillna(False).infer_objects(copy=False))
        
        event_dates = self.z_score_series[crosses].index
        
        results = []
        for event_date in event_dates:
            # We need future performance, so we must ensure we have data
            idx_loc = self.price_series.index.get_loc(event_date)
            
            # Approximate days: 3M ~ 63 days, 6M ~ 126 days, 1Y ~ 252 days
            try:
                price_0 = self.price_series.iloc[idx_loc]
                
                # Check if we have enough future data
                if idx_loc + 63 < len(self.price_series):
                    ret_3m = (self.price_series.iloc[idx_loc + 63] / price_0) - 1
                else:
                    ret_3m = np.nan
                    
                if idx_loc + 126 < len(self.price_series):
                    ret_6m = (self.price_series.iloc[idx_loc + 126] / price_0) - 1
                else:
                    ret_6m = np.nan
                    
                if idx_loc + 252 < len(self.price_series):
                    future_1y_prices = self.price_series.iloc[idx_loc:idx_loc+252]
                    max_dd_1y = ((future_1y_prices - future_1y_prices.cummax()) / future_1y_prices.cummax()).min()
                else:
                    max_dd_1y = np.nan
                    
                results.append({
                    "Date": event_date.strftime("%Y-%m-%d"),
                    "Z-Score": self.z_score_series.loc[event_date],
                    "Next 3M Return": ret_3m,
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
            "history_table": df_results.tail(5).set_index("Date"), # Show last 5 events
            "veto_printout": printout
        }
