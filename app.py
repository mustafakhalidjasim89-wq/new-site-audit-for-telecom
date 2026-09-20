import sys
import os
import io
import time
import re
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from math import radians, cos, sin, asin, sqrt

from google import genai
from google.genai import types
from google.genai.errors import APIError

from streamlit_js_eval import get_geolocation
from supabase import create_client, Client
from pypdf import PdfReader

# Optional PyMuPDF (fitz) for rendering PDF pages as resized images
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# Optional PyZBar for barcode/QR reading
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

# ---------------------------------------------------------
# 0. Path Resolution & Setup
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from kml_parser import parse_telecom_kml
except ImportError:
    def parse_telecom_kml(path):
        return []

# ---------------------------------------------------------
# Helper: Aggressive Image Optimization
# ---------------------------------------------------------
def optimize_image(uploaded_file, max_size=(800, 800), quality=75):
    """Downscales and compresses images to lower bandwidth usage."""
    img = Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer)

# ---------------------------------------------------------
# Helper: PDF Text & Resized Image Extractor
# ---------------------------------------------------------
def process_and_resize_pdf(pdf_file, max_chars=6000, target_dpi=100, max_size=(800, 800)):
    """
    Extracts text and converts scanned PDF pages into downscaled JPEG images.
    Drastically decreases payload size and execution latency.
    """
    text_content = ""
    resized_images = []

    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)

        # 1. Extract and clean text using PyPDF
        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                raw_text += t + "\n"

        text_content = re.sub(r'\s+', ' ', raw_text).strip()
        if len(text_content) > max_chars:
            text_content = text_content[:max_chars] + "... [TRUNCATED FOR SPEED]"

        # 2. Render and resize PDF pages as images if PyMuPDF is available
        if FITZ_AVAILABLE:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=target_dpi)
                img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70, optimize=True)
                buf.seek(0)
                resized_images.append(Image.open(buf))

    except Exception as e:
        st.error(f"Error processing PDF '{pdf_file.name}': {str(e)}")

    return text_content, resized_images

# ---------------------------------------------------------
# Helper: Robust JSON Output Parser & Repair Engine
# ---------------------------------------------------------
def parse_gemini_json(raw_text):
    """
    Cleans markdown formatting and repairs common JSON truncation or escaping errors.
    Prevents 'Unterminated string' crashes when Gemini outputs dense logs.
    """
    if not raw_text:
        raise ValueError("Empty response received from Gemini.")
    
    # 1. Strip Markdown standard code blocks
    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("`")

    # 2. Direct JSON Parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Extract JSON object substring via Regex
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. Attempt tail-repair for truncated strings or missing brackets
    try:
        repaired = cleaned.strip()
        repaired = re.sub(r',\s*([\]}])', r'\1', repaired)
        if not repaired.endswith("}"):
            if not repaired.endswith('"') and not repaired.endswith(']'):
                repaired += '"'
            if not repaired.endswith("}"):
                repaired += "\n}"
        return json.loads(repaired)
    except Exception:
        raise ValueError(f"Unparseable output from model: {raw_text[:120]}...")

# ---------------------------------------------------------
# Helper: Distance Calculation
# ---------------------------------------------------------
def calculate_distance_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        return 2 * asin(sqrt(a)) * 6371.0
    except (ValueError, TypeError):
        return float('inf')

# ---------------------------------------------------------
# Helper: Dynamic Model Selection and Generation
# ---------------------------------------------------------
def generate_gemini_content_robust(client, contents, config):
    configured_model = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL")
    candidate_models = []
    
    if configured_model:
        candidate_models.append(configured_model)
    
    candidate_models.extend(["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"])
    
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            return response.text
        except Exception as err:
            last_error = err
            err_msg = str(err).lower()
            if "404" in err_msg or "not_found" in err_msg or "no longer available" in err_msg:
                continue
            elif "429" in err_msg or "resource_exhausted" in err_msg:
                time.sleep(3)

    try:
        for m in client.models.list():
            if "generateContent" in getattr(m, "supported_generation_methods", []) or "flash" in m.name:
                model_id = m.name.replace("models/", "")
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=contents,
                        config=config
                    )
                    return response.text
                except Exception:
                    continue
    except Exception:
        pass

    raise last_error

# ---------------------------------------------------------
# Helper: Supabase Client Connection
# ---------------------------------------------------------
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    return create_client(url, key) if url and key else None

def save_report_to_supabase(site_id, technician, status, report_text, user_lat, user_lon):
    try:
        supabase = get_supabase_client()
        if not supabase:
            st.error("❌ Supabase URL or Key missing!")
            return False

        data = {
            "site_id": site_id,
            "technician": technician,
            "coordinates": f"{user_lat}, {user_lon}" if user_lat else "N/A",
            "status": status,
            "report_text": report_text
        }
        supabase.table("audit_reports").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"⚠️ Database error: {str(e)}")
        return False

def convert_df_to_excel(df, sheet_name='Audit_Reports'):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

# ---------------------------------------------------------
# 1. Page Config & CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telecom Audit AI",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 950px;
    }
    .stApp {
        background-color: #121417;
        color: #e2e8f0;
    }
    .header-card {
        background-color: #1e222b;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid #2d3442;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 8px 16px;
        font-weight: 600;
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
        if st.form_submit_button("Log In"):
            if USER_CREDENTIALS.get(username_input) == password_input:
                st.session_state["authenticated"] = True
                st.session_state["logged_user"] = username_input
                st.rerun()
            else:
                st.error("Invalid credentials.")
    st.stop()

# ---------------------------------------------------------
# 3. KML Loader
# ---------------------------------------------------------
KML_EXACT_PATH = os.path.join(BASE_DIR, "data", "sites.kml")

@st.cache_data(ttl=300)
def load_kml_dataset():
    if os.path.exists(KML_EXACT_PATH):
        return parse_telecom_kml(KML_EXACT_PATH)
    return []

df_sites = pd.DataFrame(load_kml_dataset())

# ---------------------------------------------------------
# 4. Header & Navigation Tabs
# ---------------------------------------------------------
st.markdown("""
    <div class='header-card'>
        <h3 style='color: #38bdf8; margin:0;'>📡 Telecom Site Audit AI (NTG Tagging)</h3>
        <p style='color: #94a3b8; margin:2px 0 0 0; font-size:12px;'>R3-BAG-CLS5 Supervisor Engine</p>
    </div>
""", unsafe_allow_html=True)

tab_audit, tab_pm = st.tabs(["🔍 Field Audit & NTG", "📄 Multi-PM Analyzer"])

# ---------------------------------------------------------
# TAB 1: Field Audit
# ---------------------------------------------------------
with tab_audit:
    col_site, col_tech = st.columns(2)
    with col_site:
        manual_site_input = st.text_input("SITE ID", placeholder="BAG0123").strip().upper()
    with col_tech:
        tech_name_input = st.text_input("TECHNICIAN", placeholder="Tech Name").strip()

    loc = get_geolocation()
    user_lat, user_lon = (loc['coords']['latitude'], loc['coords']['longitude']) if loc and 'coords' in loc else (None, None)

    is_location_valid = False
    if manual_site_input == "BAG0000":
        is_location_valid = True
        st.success("🃏 Joker Test Site Active (BAG0000). GPS validation bypassed.")
    elif manual_site_input and not df_sites.empty:
        target_col = next((c for c in ['site_code', 'site_id', 'name'] if c in df_sites.columns), None)
        if target_col:
            matched = df_sites[df_sites[target_col].astype(str).str.strip().str.upper() == manual_site_input]
            if not matched.empty:
                site_lat, site_lon = matched.iloc[0].get('latitude'), matched.iloc[0].get('longitude')
                if site_lat and site_lon and user_lat and user_lon:
                    dist_m = int(calculate_distance_km(user_lat, user_lon, site_lat, site_lon) * 1000)
                    if dist_m <= 200:
                        is_location_valid = True
                        st.success(f"✅ GPS Validated ({dist_m}m away).")
                    else:
                        st.error(f"❌ GPS Mismatch: {dist_m}m away. Must be < 200m.")
                else:
                    st.warning("⚠️ GPS signal needed for validation.")
            else:
                is_location_valid = True
        else:
            is_location_valid = True
    elif manual_site_input:
        is_location_valid = True

    if "captured_photos" not in st.session_state:
        st.session_state["captured_photos"] = []

    uploaded_files = []
    if manual_site_input and is_location_valid:
        input_mode = st.radio("Input Source:", ["Camera", "Gallery"], horizontal=True)
        if input_mode == "Camera":
            img_file = st.camera_input("Take Picture")
            if img_file and not any(p.getvalue() == img_file.getvalue() for p in st.session_state["captured_photos"]):
                st.session_state["captured_photos"].append(img_file)
