import sys
import os
import io
import time
import re
import json
import smtplib
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
def process_and_resize_pdf(pdf_file, max_chars=5000, target_dpi=100, max_size=(800, 800)):
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
        # Fix trailing commas inside arrays or objects before closing
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
# Helper: Robust Gemini Generation with Active Models
# ---------------------------------------------------------
def generate_gemini_content_robust(client, contents, config):
    configured_model = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL")
    candidate_models = []
    
    if configured_model:
        candidate_models.append(configured_model)
    
    # Active supported models
    candidate_models.extend(["gemini-3.6-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
    
    # Remove duplicates while preserving order
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
        except APIError as api_err:
            last_error = api_err
            if "404" in str(api_err) or "NOT_FOUND" in str(api_err):
                continue
            elif "429" in str(api_err) or "RESOURCE_EXHAUSTED" in str(api_err):
                time.sleep(3)
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

            if st.button("🗑️ Clear All"):
                st.session_state["captured_photos"] = []
                st.rerun()

            uploaded_files = st.session_state["captured_photos"]
        else:
            img_files = st.file_uploader("Upload photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            if img_files:
                uploaded_files.extend(img_files)

        if uploaded_files:
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    st.image(file, width=80)

    if st.button("📤 Run Audit & Submit", use_container_width=True, disabled=(not manual_site_input or not is_location_valid)):
        if not uploaded_files:
            st.warning("Please attach at least one photo.")
        else:
            gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                st.error("Missing GEMINI_API_KEY!")
            else:
                with st.spinner("⚡ Processing Audit..."):
                    try:
                        client = genai.Client(api_key=gemini_key)
                        pil_images = [optimize_image(f) for f in uploaded_files]

                        SYSTEM_PROMPT = """
You are a telecom audit engineer inspecting physical equipment, mandatory site assets, photo quality, and NTG Asset Tagging.

MANDATORY AUDIT RULES FOR EQUIPMENT & PHOTO QUALITY:
1. MANDATORY SITE EQUIPMENT CHECK: Verify presence and visual coverage of core site equipment: Diesel Generator (DG), Power Cabinet/Rectifiers, Battery Banks, Main Antenna/Tower structure, and Microwave units.
2. MISSING OR UNCLEAR PHOTO COMPLIANCE: If any mandatory equipment (e.g., Diesel Generator (DG)) is not clearly visible in the uploaded images, or if a photo is blurry/dark/partially obstructed, you MUST explicitly flag it under "MISSING OR UNCLEAR EQUIPMENT PHOTOS".
3. VERDICT RULE: If key mandatory items like DG or Rectifiers have NO clear photos attached, mark the Final Verdict as "PASS WITH CONCERNS" or "FAIL".

MANDATORY OUTPUT FORMAT:
### 1. EQUIPMENT QUANTITY COUNT & AUDIT
| Equipment / Asset Description | Identified Model / Brand | Quantities Detected | Photo Status (Clear / Blurry / Missing) |
| :--- | :--- | :--- | :--- |

### 2. MISSING OR UNCLEAR EQUIPMENT PHOTOS
* **Unclear / Poor Quality Photos:** List any equipment photos that are blurry, taken from a bad angle, or dark.
* **Missing Mandatory Photos:** Explicitly state if photos for Diesel Generator (DG), Rectifiers, Batteries, or Cables are missing.

### 3. NTG ASSET TAGGING & BARCODE VERIFICATION
* **Equipment Identifiers:** Describe NTG tags and asset labels visible in photos.
* **Cable & Port Labels:** Describe port labels/tags.

### 4. FINAL VERDICT & DEFECTS
* **Final Verdict:** [PASS / PASS WITH CONCERNS / FAIL]
* **Identified Defects:** Note trash, loose cables, unanchored items, or missing clear photo proof.
* **Corrective Actions:** Remediation steps (e.g., "Technician must re-upload a clear, full-view photo of the Diesel Generator (DG)").
"""

                        report_text = generate_gemini_content_robust(
                            client=client,
                            contents=[f"Site ID: {manual_site_input}\nTechnician: {tech_name_input or 'Unassigned'}", *pil_images],
                            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0)
                        )

                        if report_text:
                            st.subheader("📋 Audit Report")
                            st.markdown(report_text)
                            
                            status_verdict = "FAIL" if "FAIL" in report_text.upper() else ("PASS WITH CONCERNS" if "CONCERNS" in report_text.upper() else "PASS")
                            if save_report_to_supabase(manual_site_input, tech_name_input or 'Unassigned', status_verdict, report_text, user_lat, user_lon):
                                st.success("✅ Audit logged successfully!")
                                st.session_state["captured_photos"] = []
                    except Exception as e:
                        st.error(f"Audit processing failed: {str(e)}")

# ---------------------------------------------------------
# TAB 2: Multi-PM Analyzer (Batch PDF Upload & Processing)
# ---------------------------------------------------------
with tab_pm:
    st.subheader("📄 Multi-PM Checksheet Analyzer")
    
    uploaded_pdfs = st.file_uploader(
        "Upload Multiple PM PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PM checksheets simultaneously."
    )

    if uploaded_pdfs:
        st.info(f"📂 **{len(uploaded_pdfs)}** PDF file(s) loaded.")

        if st.button("🚀 Process All PM PDFs", use_container_width=True):
            gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                st.error("Missing GEMINI_API_KEY!")
            else:
                client = genai.Client(api_key=gemini_key)
                all_site_data = []

                BATCH_SYSTEM_PROMPT = """
You are a senior telecom audit supervisor analyzing PM checksheets and attached site photos. 
Examine ALL provided text and image pages carefully to extract accurate technical audit details.

STRICT DIESEL GENERATOR (DG) AUDIT RULES:
1. DIESEL GENERATOR (DG) PHOTO & SCREEN MANDATE:
   - Mandatory DG photo evidence includes: (a) Overall DG unit image, (b) DG LED Controller Display Screen photo showing running hours, and (c) Specific DG running hours numerical value in the text.
   - IF ANY OF THESE ARE MISSING OR UNCLEAR:
     * Explicitly list missing items in `missing_equipment_photos` (e.g., "DG overall photo missing", "DG LED screen photo missing", "DG running hours unrecorded").
     * State clearly in `critical_remarks`: "Site without DG / DG evidence missing - Mandatory DG photo, LED screen image, and running hours required".
     * Set the `verdict` to "REJECTED" or "APPROVED WITH CONCERNS".

2. DIESEL GENERATOR REMARKS & DEFECT WARNINGS:
   - RUNNING HOURS WARNING: Explicitly mention DG running hours value. Flag a WARNING if running hours are high or overdue for service.
   - PHYSICAL DEFECTS: State if there are oil leaks, fuel leaks, broken canopy, loose wiring, low oil pressure, or active alarms.
   - GENERAL SITE DEFECTS: Include other issues like trash/leaves inside compound, unanchored cabinets, or loose cabling.

3. SUPERVISOR FOCUS NOTES:
   - Give direct instructions to technician (e.g., "Re-upload clear DG photo and DG LED screen showing running hours", "Clean oil leakage under DG").

Return strictly valid JSON matching this structure:
{
  "site_id": "Extracted Site ID",
  "vendor_technician": "Technician Name",
  "pm_date": "YYYY-MM-DD",
  "verdict": "APPROVED" | "APPROVED WITH CONCERNS" | "REJECTED",
  "missing_equipment_photos": ["List missing or unclear photos e.g. DG Unit, DG LED Screen, Rectifier, Battery"],
  "critical_remarks": ["Detailed specific remarks: DG running hours, DG oil leaks/defects, or 'Site without DG / DG evidence missing'"],
  "supervisor_focus_notes": ["Specific corrective actions required from technician"]
}
"""

                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, pdf_file in enumerate(uploaded_pdfs):
                    status_text.text(f"⚙️ Processing ({idx+1}/{len(uploaded_pdfs)}): {pdf_file.name}")
                    
                    # Process text and extract downscaled page images
                    text_content, resized_page_images = process_and_resize_pdf(pdf_file)
                    
                    payload = [f"FILENAME: {pdf_file.name}\nEXTRACTED TEXT:\n{text_content}"]
                    if resized_page_images:
                        payload.extend(resized_page_images[:5])  # Send top 5 downscaled pages to capture complete evidence

                    # Increased max_output_tokens to 2048 to prevent JSON output truncation crashes
                    gen_config = types.GenerateContentConfig(
                        system_instruction=BATCH_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=2048,
                        response_mime_type="application/json"
                    )

                    try:
                        raw_response = generate_gemini_content_robust(
                            client=client,
                            contents=payload,
                            config=gen_config
                        )
                        
                        # Parse with robust JSON repair engine
                        parsed = parse_gemini_json(raw_response)
                        parsed["filename"] = pdf_file.name
                        all_site_data.append(parsed)
                        
                    except Exception as e:
                        all_site_data.append({
                            "filename": pdf_file.name,
                            "site_id": "ERROR",
                            "vendor_technician": "N/A",
                            "pm_date": "N/A",
                            "verdict": "REJECTED",
                            "missing_equipment_photos": ["Error parsing report"],
                            "critical_remarks": [f"Processing error: {str(e)}"],
                            "supervisor_focus_notes": ["Verify document format or JSON truncation"]
                        })

                    progress_bar.progress((idx + 1) / len(uploaded_pdfs))

                status_text.success("✅ All PM files processed successfully!")
                st.session_state["pm_analysis_results"] = all_site_data

    if "pm_analysis_results" in st.session_state and st.session_state["pm_analysis_results"]:
        results = st.session_state["pm_analysis_results"]
        st.markdown("### 📋 Multi-PM Audit Summary")
        
        summary_rows = [{
            "Site ID": r.get("site_id", "N/A"),
            "Technician": r.get("vendor_technician", "N/A"),
            "Date": r.get("pm_date", "N/A"),
            "Verdict": r.get("verdict", "N/A"),
            "Missing/Unclear Photos": " | ".join(r.get("missing_equipment_photos", [])) if isinstance(r.get("missing_equipment_photos"), list) else str(r.get("missing_equipment_photos", "")),
            "Critical Remarks": " | ".join(r.get("critical_remarks", [])) if isinstance(r.get("critical_remarks"), list) else str(r.get("critical_remarks", "")),
            "Focus Actions": " | ".join(r.get("supervisor_focus_notes", [])) if isinstance(r.get("supervisor_focus_notes"), list) else str(r.get("supervisor_focus_notes", "")),
            "Filename": r.get("filename", "")
        } for r in results]
        
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)

        st.download_button(
            label="📥 Download Multi-PM Summary (.xlsx)",
            data=convert_df_to_excel(df_summary, sheet_name='PM_Summary'),
            file_name="Multi_PM_Summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
