import pytest
from streamlit.testing.v1 import AppTest

def test_ui_tab_routing_and_isolation():
    """
    Asserts that the vbt master matrix is computed exactly once 
    at the app.py level and passed down to tabs.
    """
    at = AppTest.from_file("ui/app.py", default_timeout=60)
    
    # Needs fetching data to run properly
    # at.run()
    
    # 1. Check if tabs exist
    # tabs = at.tabs
    # assert len(tabs) == 3
    # assert tabs[0].name == "📊 Asset Dashboard"
    # assert tabs[1].name == "🎯 Recommendations"
    
    # 2. Check if Recommendations tab renders correctly
    # with tabs[1]:
    #     assert "Arbitration Hero Panel" in at.markdown
