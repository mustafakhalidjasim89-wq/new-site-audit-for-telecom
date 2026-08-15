import sys
import os
import streamlit as st
import pandas as pd
from PIL import Image
from math import radians, cos, sin, asin, sqrt
from google import genai
from streamlit_js_eval import get_geolocation

# ---------------------------------------------------------
# 0. Fix Import Paths for Streamlit Cloud Runtime
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from kml_parser import parse_telecom_kml
from geo_utils import find_nearby_sites

# ---------------------------------------------------------
# Helper: Haversine Formula for Distance (in km)
# ---------------------------------------------------------
def calculate_distance_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        r = 6371.0  # Earth's radius in kilometers
        return c * r
    except (ValueError, TypeError):
        return float('inf')

# ---------------------------------------------------------
# 1. Page Configuration & Custom Dark UI Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Site Audit AI",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #1a1d24;
        color: #ffffff;
    }
    .header-card {
        background-color: #222630;
        padding: 18px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #343a46;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
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
    st.stop()

st.sidebar.write(f"Logged in as: **{st.session_state.get('logged_user')}**")
if st.sidebar.button("Log Out"):
    st.session_state["authenticated"] = False
    st.rerun()

# ---------------------------------------------------------
# 3. KML Dataset Loader
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
# 4. App Header
# ---------------------------------------------------------
st.markdown("""
    <div class='header-card'>
        <h2 style='color: #00d2ff; margin:0;'>📡 Telecom Site Audit AI</h2>
        <p style='color: #8e9aaf; margin:4px 0 0 0;'>Designed by Mustafa Khalid / Supervisor / R3-BAG-CLS5</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. Dropdown Site Selection & Geofence Validation
# ---------------------------------------------------------
col_site, col_tech = st.columns(2)

with col_site:
    if not df_sites.empty and 'site_code' in df_sites.columns:
        site_list = sorted(df_sites['site_code'].unique().tolist())
        selected_site_code = st.selectbox("SELECT SITE ID", options=["-- Select Site --"] + site_list)
    else:
        selected_site_code = st.selectbox("SELECT SITE ID", options=["-- No Sites Found --"])

with col_tech:
    tech_name_input = st.text_input("TECHNICIAN NAME", placeholder="e.g. Alaa Fadel").strip()

# Fetch Technician Location via GPS
loc = get_geolocation()
user_lat, user_lon = None, None

if loc and 'coords' in loc:
    user_lat = loc['coords']['latitude']
    user_lon = loc['coords']['longitude']
    st.sidebar.success(f"🌐 GPS Active: {user_lat:.4f}, {user_lon:.4f}")
else:
    st.sidebar.warning("⚠️ GPS inactive. Please enable browser location permissions.")

# Distance & Geofence Check (3km Radius)
is_location_valid = False
site_data = None

if selected_site_code and selected_site_code != "-- Select Site --" and not df_sites.empty:
    matched = df_sites[df_sites['site_code'] == selected_site_code]
    if not matched.empty:
        site_data = matched.iloc[0].to_dict()
        site_lat = site_data.get('latitude')
        site_lon = site_data.get('longitude')

        if site_lat and site_lon:
            st.info(f"📍 Target Site Coordinates: Lat {site_lat}, Lon {site_lon}")
            
            if user_lat is not None and user_lon is not None:
                distance = calculate_distance_km(user_lat, user_lon, site_lat, site_lon)
                
                if distance <= 3.0:
                    is_location_valid = True
                    st.success(f"✅ GPS Match Confirmed: You are **{distance:.2f} km** from the site (Within 3 km limit).")
                else:
                    st.error(f"❌ Location Mismatch: You are **{distance:.2f} km** away from this site. You must be within **3 km** to perform an audit.")
            else:
                st.warning("⚠️ GPS Signal Required: Please turn on location services on your device to verify your proximity.")

            # Nearby sites display
            nearby = find_nearby_sites(site_lat, site_lon, raw_sites, radius_km=5.0)
            if nearby:
                with st.expander(f"📍 View {len(nearby)} Nearby Sites (Within 5km)"):
                    st.dataframe(pd.DataFrame(nearby)[['site_code', 'name', 'distance_km']], use_container_width=True)

# ---------------------------------------------------------
# 6. Photos Capture & Upload (Locked until 3km GPS Match)
# ---------------------------------------------------------
st.markdown("### PHOTOS")

uploaded_files = []

if not is_location_valid:
    st.error("🔒 Photo upload and analysis are locked. Please select a site and confirm you are within 3 km of the location.")
else:
    input_mode = st.radio("Choose Input Method:", ["Camera", "Gallery"], horizontal=True)

    if input_mode == "Camera":
        img_file = st.camera_input("Capture Site Photo")
        if img_file:
            uploaded_files.append(img_file)
    else:
        img_files = st.file_uploader("Pick from gallery", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if img_files:
            uploaded_files.extend(img_files)

    if uploaded_files:
        st.write(f"Selected Photos ({len(uploaded_files)}):")
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 4]:
                st.image(file, use_container_width=True)

# ---------------------------------------------------------
# 7. AI Vision Inspection Processing (Geofence Enforced)
# ---------------------------------------------------------
st.write("---")
if st.button("🔍 Analyze Site", use_container_width=True, disabled=not is_location_valid):
    if not uploaded_files:
        st.warning("Please capture or upload at least one site photo.")
    else:
        gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

        if not gemini_key:
            st.error("❌ Missing Gemini API Key! Please configure `GEMINI_API_KEY` in Streamlit Secrets.")
        else:
            with st.spinner("🤖 Gemini AI Vision is inspecting equipment and analyzing photos..."):
                try:
                    client = genai.Client(api_key=gemini_key)
                    pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]

                    prompt = f"""
                    You are an expert telecommunications site audit engineer inspecting field photos.
                    Site ID: {selected_site_code}
                    Technician: {tech_name_input or 'Unassigned'}

                    Analyze the provided image(s) thoroughly and generate a structured site inspection report:
                    1. **Equipment Identified**: Cabinets (Huawei/ETP48), Rectifiers, Antennas, RRUs, Microwave transmission dishes, Lithium/Lead-Acid Batteries, Solar installations.
                    2. **Installation Quality & Cabling**: Cable routing neatness, grounding status, physical damage, cleanliness.
                    3. **Defects & Safety Hazards**: Uncapped or exposed cables, water ingress signs, burnt connectors, loose mountings.
                    4. **Final Verdict**: PASS, PASS WITH CONCERNS, or FAIL (Include justification and required corrective actions).
                    """

                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[prompt, *pil_images]
                    )

                    st.success(f"✅ Audit completed for Site **{selected_site_code}**!")
                    st.markdown("### 📋 AI Audit Analysis Report")
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"⚠️ AI Vision Analysis failed: {str(e)}")
