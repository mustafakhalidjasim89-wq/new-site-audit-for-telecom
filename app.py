import os
import streamlit as st
import pandas as pd
from kml_parser import parse_telecom_kml

# Dynamic base directory matching your screenshot structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KML_PATH = os.path.join(BASE_DIR, "data", "sites.kml")

@st.cache_data
def load_sites():
    if os.path.exists(KML_PATH):
        return parse_telecom_kml(KML_PATH)
    return []

sites_data = load_sites()
