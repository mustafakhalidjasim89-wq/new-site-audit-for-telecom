import sys
import os
import streamlit as st
import pandas as pd
from PIL import Image
from google import genai

# ---------------------------------------------------------
# 0. Fix Import Paths for Streamlit Cloud Runtime
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from kml_parser import parse_telecom_kml
from geo_utils import find_nearby_sites

# ---------------------------------------------------------
# 1. Page Configuration & Custom Dark UI Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Site Audit AI",
    page_icon="📡",
    layout="wide"
)

# Dark glassmorphism theme
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

# Sidebar user session management
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
# 5. Site Identification & Field Inputs
# ---------------------------------------------------------
col_site, col_tech = st.columns(2)

with col_site:
    site_id_input = st.text_input("SITE ID", placeholder="e.g. IQ-BG-1042 or BAG6436").strip().upper()

with col_tech:
    tech_name_input = st.text_input("TECHNICIAN NAME", placeholder="e.g. Alaa Fadel").strip()

# KML Coordinates Match & Nearby Sites Lookup
if site_id_input and not df_sites.empty:
    matched = df_sites[df_sites['site_code'].str.upper() == site_id_input]
    if not matched.empty:
        site_data = matched.iloc[0].to_dict()
        st.success(f"📍 Location Found: Latitude {site_data.get('latitude')}, Longitude {site_data.get('longitude')}")
        
        if site_data.get('latitude') and site_data.get('longitude'):
            nearby = find_nearby_sites(site_data['latitude'], site_data['longitude'], raw_sites, radius_km=5.0)
            if nearby:
                with st.expander(f"📍 View {len(nearby)} Nearby Sites (Within 5km)"):
                    st.dataframe(pd.DataFrame(nearby)[['site_code', 'name', 'distance_km']], use_container_width=True)

# ---------------------------------------------------------
# 6. Photos Capture & Upload
# ---------------------------------------------------------
st.markdown("### PHOTOS")

input_mode = st.radio("Choose Input Method:", ["Camera", "Gallery"], horizontal=True)

uploaded_files = []
if input_mode == "Camera":
    img_file = st.camera_input("Capture Site Photo")
    if img_file:
        uploaded_files.append(img_file)
else:
    img_files = st.file_uploader("Pick from gallery", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if img_files:
        uploaded_files.extend(img_files)

# Preview uploaded site images
if uploaded_files:
    st.write(f"Selected Photos ({len(uploaded_files)}):")
    cols = st.columns(min(len(uploaded_files), 4))
    for idx, file in enumerate(uploaded_files):
        with cols[idx % 4]:
            st.image(file, use_container_width=True)

# ---------------------------------------------------------
# 7. AI Vision Inspection Processing (Google GenAI SDK)
# ---------------------------------------------------------
st.write("---")
if st.button("🔍 Analyze Site", use_container_width=True):
    if not site_id_input:
        st.warning("Please enter a SITE ID before analyzing.")
    elif not uploaded_files:
        st.warning("Please capture or upload at least one site photo.")
    else:
        gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

        if not gemini_key:
            st.error("❌ Missing Gemini API Key! Please configure `GEMINI_API_KEY` in Streamlit Secrets.")
        else:
            with st.spinner("🤖 Gemini AI Vision is inspecting equipment and analyzing photos..."):
                try:
                    # Initialize Google GenAI Client
                    client = genai.Client(api_key=gemini_key)

                    # Prepare PIL Image instances
                    pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]

                    prompt = f"""
                    You are an expert telecommunications site audit engineer inspecting field photos.
                    Site ID: {site_id_input}
                    Technician: {tech_name_input or 'Unassigned'}

                    Analyze the provided image(s) thoroughly and generate a structured site inspection report:
                    1. **Equipment Identified**: Cabinets (Huawei/ETP48), Rectifiers, Antennas, RRUs, Microwave transmission dishes, Lithium/Lead-Acid Batteries, Solar installations.
                    2. **Installation Quality & Cabling**: Cable routing neatness, grounding status, physical damage, cleanliness.
                    3. **Defects & Safety Hazards**: Uncapped or exposed cables, water ingress signs, burnt connectors, loose mountings.
                    4. **Final Verdict**: PASS, PASS WITH CONCERNS, or FAIL (Include justification and required corrective actions).
                    """

                    # Call generation endpoint using supported model string
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[prompt, *pil_images]
                    )

                    st.success(f"✅ Audit completed for Site **{site_id_input}**!")
                    st.markdown("### 📋 AI Audit Analysis Report")
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"⚠️ AI Vision Analysis failed: {str(e)}")
