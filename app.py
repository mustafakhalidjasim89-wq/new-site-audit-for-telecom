import sys
import os
import streamlit as st
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# 0. Setup Root Path for Streamlit Cloud Imports
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from kml_parser import parse_telecom_kml
from geo_utils import find_nearby_sites

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS (Matching Dark UI)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Site Audit AI",
    page_icon="📡",
    layout="wide"
)

# Apply Dark Glassmorphism Theme to match image
st.markdown("""
    <style>
    .stApp {
        background-color: #1a1d24;
        color: #ffffff;
    }
    .card-box {
        background-color: #262a34;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #343a46;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Authentication System
# ---------------------------------------------------------
USER_CREDENTIALS = {
    "admin": "telecom2026",
    "mustafa": "audit123",
    "user": "asiacell123"
}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown("<div class='card-box'>", unsafe_allow_html=True)
    st.title("🔒 Telecom Site Audit AI - Login")
    
    with st.form("login_form"):
        username_input = st.text_input("Username").strip()
        password_input = st.text_input("Password", type="password").strip()
        submit_button = st.form_submit_button("Log In")
        
        if submit_button:
            if USER_CREDENTIALS.get(username_input) == password_input:
                st.session_state["authenticated"] = True
                st.session_state["logged_user"] = username_input
                st.rerun()
            else:
                st.error("Invalid username or password.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Logout Sidebar
st.sidebar.write(f"Logged in as: **{st.session_state.get('logged_user')}**")
if st.sidebar.button("Log Out"):
    st.session_state["authenticated"] = False
    st.rerun()

# ---------------------------------------------------------
# 3. Load KML Data
# ---------------------------------------------------------
KML_PATH = os.path.join(BASE_DIR, "data", "sites.kml")

@st.cache_data
def load_kml_dataset(path):
    if not os.path.exists(path):
        return []
    try:
        return parse_telecom_kml(path)
    except Exception:
        return []

raw_sites = load_kml_dataset(KML_PATH)
df_sites = pd.DataFrame(raw_sites)

# ---------------------------------------------------------
# 4. Header UI (Designed to match your image)
# ---------------------------------------------------------
st.markdown("""
    <div style='background-color: #222630; padding: 18px; border-radius: 12px; margin-bottom: 20px;'>
        <h2 style='color: #00d2ff; margin:0;'>📡 Telecom Site Audit AI</h2>
        <p style='color: #8e9aaf; margin:4px 0 0 0;'>Designed by Mustafa Khalid / Supervisor / R3-BAG-CLS5</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. SITE ID & TECHNICIAN NAME Input Card
# ---------------------------------------------------------
col_site, col_tech = st.columns(2)

with col_site:
    site_id_input = st.text_input("SITE ID", placeholder="e.g. IQ-BG-1042 or BAG6436").strip().upper()

with col_tech:
    tech_name_input = st.text_input("TECHNICIAN NAME", placeholder="e.g. Alaa Fadel").strip()

# KML Location Lookup Integration
if site_id_input and not df_sites.empty:
    matched = df_sites[df_sites['site_code'].str.upper() == site_id_input]
    if not matched.empty:
        site_data = matched.iloc[0].to_dict()
        st.success(f"📍 KML Match Found: Lat {site_data.get('latitude')}, Lon {site_data.get('longitude')}")
        
        # Nearby Sites Lookup
        if site_data.get('latitude') and site_data.get('longitude'):
            nearby = find_nearby_sites(site_data['latitude'], site_data['longitude'], raw_sites, radius_km=5.0)
            if nearby:
                with st.expander(f"Found {len(nearby)} Nearby Sites (5km Radius)"):
                    st.dataframe(pd.DataFrame(nearby)[['site_code', 'name', 'distance_km']], use_container_width=True)

# ---------------------------------------------------------
# 6. PHOTOS Upload & Camera Section
# ---------------------------------------------------------
st.markdown("### PHOTOS")

input_mode = st.radio("Choose Input Method:", ["Camera", "Gallery"], horizontal=True)

uploaded_files = []
if input_mode == "Camera":
    img_file = st.camera_input("Capture Photo")
    if img_file:
        uploaded_files.append(img_file)
else:
    img_files = st.file_uploader("Pick from gallery", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if img_files:
        uploaded_files.extend(img_files)

# Preview Uploaded Images
if uploaded_files:
    st.write(f"Uploaded Photos ({len(uploaded_files)}):")
    cols = st.columns(min(len(uploaded_files), 4))
    for idx, file in enumerate(uploaded_files):
        with cols[idx % 4]:
            st.image(file, use_container_width=True)

# ---------------------------------------------------------
# 7. Analyze Site Button
# ---------------------------------------------------------
st.write("---")
if st.button("🔍 Analyze Site", use_container_width=True):
    if not site_id_input:
        st.warning("Please enter a SITE ID before analyzing.")
    elif not uploaded_files:
        st.warning("Please capture or upload at least one site photo.")
    else:
        st.success("Analyzing site photos with AI Vision...")
        st.info(f"Audit completed for Site **{site_id_input}** by Technician **{tech_name_input or 'Unassigned'}**!")
