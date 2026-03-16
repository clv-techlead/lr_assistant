# app.py
# This is the entry point for the Streamlit app.
# Run it with: streamlit run app.py
# It sets up the page layout and routes each tab to its own file.

import streamlit as st

# Import the render function from our Tab 1 file
# We'll add tabs 2, 3, 4 here as we build them
from tabs.ca_interpreter import render as render_ca_interpreter

# --- Page configuration ---
# This must be the first Streamlit command in the file
st.set_page_config(
    page_title="LR Operations Assistant",
    page_icon="⚖️",
    layout="wide"  # Use full browser width — better for a professional tool
)

# --- App header ---
st.title("⚖️ LR Operations Assistant")
st.caption("A Labour Relations productivity tool for TDSB — built to support consistent, efficient, and well-grounded LR practice.")

st.divider()

# --- Tab layout ---
# Each tab maps to one of our four features
# st.tabs returns a list of tab objects — we unpack them into named variables
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 CA Interpreter",
    "📋 Grievance Prep",
    "🎓 Training Generator",
    "📊 Trend Dashboard"
])

# Tab 1 — fully built
with tab1:
    render_ca_interpreter()

# Tabs 2–4 — placeholders until we build them
with tab2:
    st.info("🚧 Grievance Prep Assistant — coming soon.")

with tab3:
    st.info("🚧 Training Material Generator — coming soon.")

with tab4:
    st.info("🚧 Trend Dashboard — coming soon.")
