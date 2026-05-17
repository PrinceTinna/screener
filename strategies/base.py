from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.
    Ensures a consistent interface for the UI and Optimizer.
    """
    
    @property
    @abstractmethod
    def name(self):
        """Returns the human-readable name of the strategy."""
        pass

    @property
    @abstractmethod
    def description(self):
        """Returns a short description of the strategy logic."""
        pass

    @abstractmethod
    def run(self, data: pd.DataFrame, params: dict):
        """
        Executes the strategy logic and returns a result object
        containing entries, exits, and any intermediate indicators.
        """
        pass

    @abstractmethod
    def get_default_params(self):
        """Returns a dictionary of default parameters."""
        pass
