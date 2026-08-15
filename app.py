import streamlit as st
import pandas as pd
import os
from kml_parser import parse_telecom_kml
from geo_utils import find_nearby_sites

# Configure page
st.set_page_config(page_title="Telecom Site Audit", layout="wide")

# 1. Load KML Data efficiently
@st.cache_data
def load_kml_data():
    kml_path = os.path.join("data", "sites.kml")
    return parse_telecom_kml(kml_path)

sites_data = load_kml_data()
df_sites = pd.DataFrame(sites_data)

st.title("📡 Telecom Site Audit & Nearby Search")

# 2. Search for a specific site
search_code = st.text_input("Enter Site Code (e.g., BAG6436):").strip().upper()

if search_code:
    # Filter dataframe for the entered site code
    target_site = df_sites[df_sites['site_code'].str.upper() == search_code]

    if not target_site.empty:
        site_info = target_site.iloc[0]
        st.success(f"Site {site_info['site_code']} Found!")
        
        # Display Site Details
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", site_info['latitude'])
        with col2:
            st.metric("Longitude", site_info['longitude'])

        # 3. Use geo_utils to find nearby sites
        st.subheader("Nearby Sites (5km Radius)")
        
        # Convert list of dicts to the format expected by your geo_utils
        nearby = find_nearby_sites(
            lat=site_info['latitude'], 
            lon=site_info['longitude'], 
            all_sites=sites_data, 
            radius_km=5.0
        )
        
        if nearby:
            nearby_df = pd.DataFrame(nearby)
            st.dataframe(nearby_df[['site_code', 'name', 'distance_km']])
            
            # 4. Show on Map
            st.map(nearby_df[['latitude', 'longitude']])
        else:
            st.info("No nearby sites found within radius.")
            st.map(target_site[['latitude', 'longitude']])

    else:
        st.error(f"Site '{search_code}' not found in KML database.")

# Optional: Show full raw dataset
with st.expander("View Full KML Dataset"):
    st.dataframe(df_sites)
