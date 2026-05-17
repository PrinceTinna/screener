import os
import json
import pandas as pd
import streamlit as st

DATA_DIR = "data"
UNIVERSES_JSON = os.path.join(DATA_DIR, "universes.json")

def get_universe_tickers(category):
    """
    Returns lists for standard NSE indices by reading from data/universes.json.
    Falls back to building it if missing.
    """
    if not os.path.exists(UNIVERSES_JSON):
        # Initial empty state if not yet synced
        loaded_universes = {}
    else:
        with open(UNIVERSES_JSON, 'r') as f:
            loaded_universes = json.load(f)
            
    custom_registry = {
        'My Portfolio': [
            'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
            'ICICIGI.NS', 'IDFCFIRSTB.NS', 'MOTILALOFS.NS', 'NUVAMA.NS',
            'KFINTECH.NS', 'PRUDENT.NS', 'PNBHOUSING.NS',
            'TECHM.NS', 'LTTS.NS', 'KPITTECH.NS', 'CYIENT.NS', 'TATATECH.NS',
            'TATACOMM.NS', 'AFFLE.NS', 'INFOBEAN.NS', 'NAZARA.NS',
            'OPTIEMUS.NS', 'KAYNES.NS', 'WOCKPHARMA.NS', 'SHILPAMED.NS', 
            'AARTIPHARM.NS', 'MARKSANS.NS', 'PPLPHARMA.NS', 'PIIND.NS',
            'ITC.NS', 'MARICO.NS', 'NESTLEIND.NS', 'GODREJCP.NS',
            'TATACONSUM.NS', 'UNITDSPR.NS', 'GILLETTE.NS', 'BATAINDIA.NS',
            'M&M.NS', 'CGPOWER.NS', 'MINDACORP.NS', 'AMBER.NS', 'EPACK.NS',
            'SYRMA.NS', 'MAHLIFE.NS', 'ABFRL.NS', 'SIYSIL.NS',
            'RELIANCE.NS', 'COALINDIA.NS', 'PCBL.NS', 'LXCHEM.NS',
            'TATACHEM.NS', 'RALLIS.NS', 'ZYDUSWELL.NS',
            'GRASIM.NS', 'KAJARIACER.NS', 'CERA.NS', 'JUBLINGREA.NS',
            'SUNTECK.NS', 'DELHIVERY.NS', 'RPSGVENT.NS', 'INDIGRID.NS',
            'TATAELXSI.NS', 'OLECTRA.NS', 'OLAELEC.NS', 'ETERNAL.NS',
            'ATL.NS', 'HLEGLAS.NS', 'FIVESTAR.NS', 'LODHA.NS', 'STARHEALTH.NS'
        ],
        'Test 3 Asset Universe': ['HDFCBANK.NS', 'BAJFINANCE.NS', 'RELIANCE.NS'],
        'Index-ETF': [
            'CPSEETF.NS', 'KOTAKBKETF.NS', 'SHARIABEES.NS', 'NIFTYBEES.NS', 
            'JUNIORBEES.NS', 'SETFNIF50.NS', 'KOTAKNIFTY.NS', 'GOLDBEES.NS', 
            'SETFGOLD.NS', 'KOTAKGOLD.NS', 'LIQUIDBEES.NS', '^NSEI', '^BSESN', 
            '^CRSLDX', '^CRSLMD', '^NSEBANK', '^CNXIT', '^CNXFMCG', '^CNXPHARMA', '^CNXAUTO'
        ]
    }
    
    registry = {**loaded_universes, **custom_registry}

    index_symbols = {
        'NIFTY 50': '^NSEI',
        'NIFTY NEXT 50': '^NSMIDCP',
        'NIFTY BANK': '^NSEBANK',
        'NIFTY IT': 'CNXIT.NS',
        'NIFTY PHARMA': '^CNXPHARMA',
        'NIFTY HEALTHCARE': '^NIFTYHEALTH',
        'NIFTY OIL & GAS': '^NIFTYOILGAS',
        'NIFTY 500': '^CRSLDX',
        'NIFTY MIDCAP 150': 'NIFTYMIDCAP150.NS',
        'NIFTY 250': 'NIFTYLARGEMID250.NS',
        'NIFTY SMALLCAP 100': '^CNXSC'
    }
    
    result = registry.get(category, [])
    clean_req = []
    if category in index_symbols:
        clean_req.append(index_symbols[category])
        
    for r in result:
        if r not in clean_req:
            clean_req.append(r)
            
    return clean_req

def fetch_and_cache_universes():
    """
    Harvests index components from NSE (Placeholder/Original Logic).
    """
    # This logic was in core/universe_fetcher.py in the old project
    # For now, let's just create a shell or move the logic here if I had it.
    # From app.py, I know it calls fetch_and_cache_universes().
    pass
