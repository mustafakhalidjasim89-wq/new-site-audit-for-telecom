import sys
import os

# ---------------------------------------------------------
# 0. Fix Import Paths for Streamlit Cloud Runtime
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd

# Local imports
from kml_parser import parse_telecom_kml
from geo_utils import find_nearby_sites

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Site Audit & Search",
    page_icon="📡",
    layout="wide"
)

# ---------------------------------------------------------
# 2. Path & Data Loading (Cached for performance)
# ---------------------------------------------------------
KML_PATH = os.path.join(BASE_DIR, "data", "sites.kml")

@st.cache_data
def load_kml_dataset(path):
    if not os.path.exists(path):
        st.error(f"KML file not found at path: {path}")
        return []
    try:
        return parse_telecom_kml(path)
    except Exception as e:
        st.error(f"Error parsing KML file: {e}")
        return []

# Load site data
raw_sites = load_kml_dataset(KML_PATH)
df_sites = pd.DataFrame(raw_sites)

# ---------------------------------------------------------
# 3. Main Interface & Title
# ---------------------------------------------------------
st.title("📡 Telecom Site Audit & Nearby Search")

if df_sites.empty:
    st.warning("No sites loaded. Please check that `data/sites.kml` exists and contains valid Placemarks.")
    st.stop()

st.sidebar.header("Dataset Overview")
st.sidebar.metric("Total Sites Loaded", len(df_sites))

# ---------------------------------------------------------
# 4. Search & Audit Section
# ---------------------------------------------------------
st.subheader("🔍 Search Site Code")

# Text input for site lookup
search_input = st.text_input("Enter Site Code / ID (e.g., BAG6436):", "").strip().upper()

if search_input:
    # Filter dataset for site code matching
    matched_rows = df_sites[df_sites['site_code'].str.upper() == search_input]

    if not matched_rows.empty:
        site_info = matched_rows.iloc[0].to_dict()
        st.success(f"Site **{site_info['site_code']}** found!")

        # Display Key Site Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Site Code", site_info.get('site_code', 'N/A'))
        with col2:
            st.metric("Latitude", site_info.get('latitude', 'N/A'))
        with col3:
            st.metric("Longitude", site_info.get('longitude', 'N/A'))

        # ---------------------------------------------------------
        # 5. Nearby Site Filtering & Geospatial Analysis
        # ---------------------------------------------------------
        st.write("---")
        st.subheader("📍 Nearby Sites Analysis")

        radius_km = st.slider("Select search radius (km):", min_value=1.0, max_value=20.0, value=5.0, step=0.5)

        if site_info.get('latitude') and site_info.get('longitude'):
            nearby_sites = find_nearby_sites(
                lat=site_info['latitude'],
                lon=site_info['longitude'],
                all_sites=raw_sites,
                radius_km=radius_km
            )

            if nearby_sites:
                df_nearby = pd.DataFrame(nearby_sites)
                
                st.write(f"Found **{len(df_nearby)}** nearby site(s) within {radius_km} km:")
                st.dataframe(df_nearby[['site_code', 'name', 'latitude', 'longitude', 'distance_km']], use_container_width=True)

                # Show Map with Target + Nearby Sites
                st.subheader("🗺️ Map View")
                map_df = df_nearby[['latitude', 'longitude']].dropna()
                st.map(map_df)
            else:
                st.info(f"No nearby sites found within a {radius_km} km radius.")
                st.map(pd.DataFrame([site_info])[['latitude', 'longitude']])
        else:
            st.warning("Coordinates unavailable for this site code.")
    else:
        st.error(f"Site Code **{search_input}** not found in the KML dataset.")

# ---------------------------------------------------------
# 6. Raw Data Explorer Expander
# ---------------------------------------------------------
with st.expander("📋 View Full Master Site List"):
    st.dataframe(df_sites, use_container_width=True)
