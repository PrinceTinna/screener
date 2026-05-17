import streamlit as st

def inject_premium_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Outfit', sans-serif; font-size: 1.15rem; }
        .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
        .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: white; border: none; }
        .status-card { padding: 20px; border-radius: 15px; background: #111; border: 1px solid #333; margin-bottom: 20px; }
        
        /* Screener Table Styles */
        .screener-row { border-radius: 8px; padding: 6px 10px; margin-bottom: 4px; border: 1px solid rgba(255,255,255,0.05); }
        .trophy-badge { font-size:0.75rem; font-weight:700; color:#1a1a1a; background:rgba(255,215,0,0.85); padding:2px 8px; border-radius:12px; border:1px solid rgba(218,165,32,0.8); margin-right:8px; }
        .best-setup-tag { font-size:0.75rem; font-weight:700; color:#ecf0f1; background:rgba(155,89,182,0.85); padding:2px 8px; border-radius:12px; border:1px solid rgba(142,68,173,0.8); margin-right:8px; }
        .active-tag { font-size:0.75rem; font-weight:700; color:#00E676; background:rgba(0,230,118,0.15); padding:2px 8px; border-radius:12px; border:1px solid rgba(0,230,118,0.5); }
        </style>
    """, unsafe_allow_html=True)
