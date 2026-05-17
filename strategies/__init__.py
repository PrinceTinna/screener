import os
import importlib
import inspect
from strategies.base import BaseStrategy

def discover_strategies():
    """
    Dynamically discovers all strategy classes in the 'strategies' directory 
    that inherit from BaseStrategy.
    """
    strategies = {}
    current_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename != "base.py" and not filename.startswith("__"):
            module_name = f"strategies.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                        strat_instance = obj()
                        strategies[strat_instance.name] = obj
            except Exception as e:
                print(f"Error loading strategy from {module_name}: {e}")
                
    return strategies

# Initialize the registry
STRATEGY_REGISTRY = discover_strategies()

def get_strategy(name):
    """Returns the strategy class for a given name."""
    return STRATEGY_REGISTRY.get(name)

def list_strategies():
    """Returns a list of discovered strategy names."""
    return list(STRATEGY_REGISTRY.keys())
