import streamlit as st
import os

# Set page parameters
st.set_page_config(
    page_title="Telecom Site Audit AI",
    page_icon="🛈",
    layout="centered"
)

# Apply Custom CSS to match the exact dark UI card design
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0f172a;
    }
    
    /* Header Container styling */
    .header-box {
        background-color: #1e293b;
        padding: 18px 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .header-title {
        color: #06b6d4;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 4px;
    }

    /* Section Containers */
    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    
    .card-box {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    
    /* Customizing Inputs */
    .stTextInput > div > div > input {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background-color: #0284c7;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 12px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #0369a1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
<div class="header-box">
    <div class="header-title">🛈 Telecom Site Audit AI</div>
    <div class="header-subtitle">Designed by Mustafa Khalid / Supervisor / R3-BAG-CLS5</div>
</div>
""", unsafe_allow_html=True)

# --- INPUT FIELDS SECTION ---
st.markdown('<div class="card-box">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    site_id = st.text_input("SITE ID", placeholder="e.g. IQ-BG-1042")

with col2:
    tech_name = st.text_input("TECHNICIAN NAME", placeholder="e.g. Alaa Fadel")
st.markdown('</div>', unsafe_allow_html=True)

# --- PHOTOS SECTION ---
st.markdown('<div class="card-box">', unsafe_allow_html=True)
st.write("**PHOTOS (0)**")

tab_cam, tab_gal = st.tabs(["📷 Camera", "📁 Gallery"])

with tab_cam:
    cam_photo = st.camera_input("Capture photos directly")

with tab_gal:
    uploaded_files = st.file_uploader("Pick photos from gallery", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if not cam_photo and not uploaded_files:
    st.info("No images uploaded. Capture photos or pick from gallery.")

st.markdown('</div>', unsafe_allow_html=True)

# --- ANALYZE SITE BUTTON ---
if st.button("🔍 Analyze Site"):
    if not site_id:
        st.warning("Please enter a SITE ID before analyzing.")
    else:
        st.success(f"Processing analysis for Site: {site_id} (Technician: {tech_name or 'N/A'})")
