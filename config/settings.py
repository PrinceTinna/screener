from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

# Ensure cache dir exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Math Constants
TRADING_DAYS_PER_YEAR = 252

# Data Fetching Constants
FETCH_START_YEAR = 2001

# UI Constants
STREAMLIT_TITLE = "QuantPro - Index Intelligence"
STREAMLIT_LAYOUT = "wide"
