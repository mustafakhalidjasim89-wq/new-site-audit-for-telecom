import sys
import os
import io
import time
import re
import json
import base64

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from math import radians, cos, sin, asin, sqrt

from google import genai
from google.genai import types
from openai import OpenAI

from streamlit_js_eval import get_geolocation
from supabase import create_client, Client
from pypdf import PdfReader

# PyMuPDF Import
try:
    import pymupdf as fitz
    FITZ_AVAILABLE = True
except ImportError:
    try:
        import fitz
        FITZ_AVAILABLE = True
    except ImportError:
        FITZ_AVAILABLE = False

# ---------------------------------------------------------
# 0. Path Resolution & KML Setup
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
# Persistent Client Initializations
# ---------------------------------------------------------
@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120000)
    )

@st.cache_resource
def get_nvidia_client(api_key):
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=120.0
    )

# ---------------------------------------------------------
# Helper Functions: Image Processing & Stitching
# ---------------------------------------------------------
def optimize_image(uploaded_file, max_size=(800, 800), quality=70):
    img = Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer)

def pil_to_bytes(pil_img, quality=70):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()

def pil_to_base64(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def stitch_images_to_grid(pil_images, cols=2, max_dim=1024):
    """
    Merges multiple PIL images into a single grid image to overcome 
    single-image prompt limits enforced by endpoints like NVIDIA NIM.
    """
    if not pil_images:
        return None
    if len(pil_images) == 1:
        return pil_images[0]

    # Calculate grid dimensions
    n_images = len(pil_images)
    rows = (n_images + cols - 1) // cols

    cell_w = max(img.width for img in pil_images)
    cell_h = max(img.height for img in pil_images)

    grid_w = cell_w * cols
    grid_h = cell_h * rows

    grid_img = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))

    for idx, img in enumerate(pil_images):
        r = idx // cols
        c = idx % cols
        grid_img.paste(img, (c * cell_w, r * cell_h))

    grid_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return grid_img

def process_and_resize_pdf(pdf_file, max_chars=4000, target_dpi=150, max_size=(1024, 1024)):
    text_content = ""
    page_images = []

    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)

        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                raw_text += t + "\n"

        text_content = re.sub(r'\s+', ' ', raw_text).strip()
        if len(text_content) > max_chars:
            text_content = text_content[:max_chars] + "... [TRUNCATED]"

        if FITZ_AVAILABLE:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=target_dpi)
                img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75, optimize=True)
                buf.seek(0)
                page_images.append(Image.open(buf))

    except Exception as e:
        st.error(f"Error processing PDF '{pdf_file.name}': {str(e)}")

    return text_content, page_images

# ---------------------------------------------------------
# Helper: JSON Repair & Parsing
# ---------------------------------------------------------
def parse_model_json(raw_text):
    if not raw_text:
        raise ValueError("Empty response received from model.")
    
    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("`")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    try:
        repaired = cleaned.strip()
        repaired = re.sub(r',\s*([\]}])', r'\1', repaired)
        if repaired.count('"') % 2 != 0:
            repaired += '"'
        open_brackets = repaired.count('[') - repaired.count(']')
        if open_brackets > 0:
            repaired += ']' * open_brackets
        open_braces = repaired.count('{') - repaired.count('}')
        if open_braces > 0:
            repaired += '}' * open_braces

        return json.loads(repaired)
    except Exception:
        raise ValueError(f"Unparseable output from model: {raw_text[:120]}...")

def calculate_distance_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        return 2 * asin(sqrt(a)) * 6371.0
    except (ValueError, TypeError):
        return float('inf')

# ---------------------------------------------------------
# AI Execution Pipelines
# ---------------------------------------------------------
def run_nemotron_pdf_parser(pdf_page_images):
    """
    Parses PDF pages via nvidia/nemotron-parse-2.0.
    Executes page-by-page to comply with the 1-image per call constraint.
    """
    nvidia_key = st.secrets.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not nvidia_key:
        raise ValueError("Missing NVIDIA_API_KEY for nemotron-parse-2.0!")

    client = get_nvidia_client(nvidia_key)
    extracted_markdown_pages = []

    for page_num, img in enumerate(pdf_page_images):
        b64_str = pil_to_base64(img)
        
        # Exactly 1 image_url payload per call
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all document text, fields, checkboxes, site ID, technician name, and tabular checksheets accurately into standard Markdown format."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
            ]
        }]

        response = client.chat.completions.create(
            model="nvidia/nemotron-parse-2.0",
            messages=messages,
            temperature=0.0,
            max_tokens=4096
        )
        
        page_md = response.choices[0].message.content
        extracted_markdown_pages.append(f"--- PAGE {page_num + 1} ---\n{page_md}")

    return "\n\n".join(extracted_markdown_pages)

def run_ai_audit_call(provider, prompt, text_content, pil_images=None, is_json=False):
    pil_images = pil_images or []

    if provider == "Google Gemini (Recommended)":
        gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("Missing GEMINI_API_KEY!")
        
        client = get_gemini_client(gemini_key)
        configured_model = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
        
        contents_payload = [f"{prompt}\n\nDOCUMENT TEXT / CONTEXT:\n{text_content}"]
        # Gemini handles native multi-image arrays
        for img in pil_images:
            img_bytes = pil_to_bytes(img)
            contents_payload.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            )

        gen_config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
            response_mime_type="application/json" if is_json else "text/plain"
        )
        
        response = client.models.generate_content(
            model=configured_model,
            contents=contents_payload,
            config=gen_config
        )
        return response.text

    elif provider == "NVIDIA (DeepSeek V4.1)":
        nvidia_key = st.secrets.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not nvidia_key:
            raise ValueError("Missing NVIDIA_API_KEY!")
        
        client = get_nvidia_client(nvidia_key)
        
        # Stitch multiple images into a 1-grid image if multiple photos are supplied
        user_content = [{"type": "text", "text": f"{prompt}\n\nDOCUMENT DATA:\n{text_content}"}]
        
        if pil_images:
            stitched_img = stitch_images_to_grid(pil_images)
            b64_str = pil_to_base64(stitched_img)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
            })

        messages_payload = [{"role": "user", "content": user_content}]

        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4.1-flash",
            messages=messages_payload,
            temperature=0.1,
            max_tokens=4096
        )
        return response.choices[0].message.content.strip()

# ---------------------------------------------------------
# Helper: Supabase Client & File Export
# ---------------------------------------------------------
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    return create_client(url, key) if url and key else None

def save_report_to_supabase(site_id, technician, status, report_text, user_lat, user_lon):
    try:
        supabase = get_supabase_client()
        if not supabase:
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
    page_title="Telecom Audit AI Engine",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; max-width: 950px; }
.stApp { background-color: #121417; color: #e2e8f0; }
.header-card {
    background-color: #1e222b;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 12px;
    border: 1px solid #2d3442;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Authentication & Sidebar Settings
# ---------------------------------------------------------
USER_CREDENTIALS = {"admin": "telecom2026", "mustafa": "audit123", "user": "asiacell123"}

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

st.sidebar.title("⚙️ AI Pipeline Selection")
ai_provider = st.sidebar.selectbox(
    "Visual Reasoning Engine",
    ["Google Gemini (Recommended)", "NVIDIA (DeepSeek V4.1)"]
)

st.sidebar.info("📄 PDF Parse Model: **NVIDIA nemotron-parse-2.0** (Active - Single-Page Batch Enabled)")

# ---------------------------------------------------------
# 3. KML Dataset Loading
# ---------------------------------------------------------
KML_EXACT_PATH = os.path.join(BASE_DIR, "data", "sites.kml")

@st.cache_data(ttl=300)
def load_kml_dataset():
    if os.path.exists(KML_EXACT_PATH):
        return parse_telecom_kml(KML_EXACT_PATH)
    return []

df_sites = pd.DataFrame(load_kml_dataset())

# ---------------------------------------------------------
# 4. Interface Header & Navigation Tabs
# ---------------------------------------------------------
st.markdown("""
<div class='header-card'>
    <h3 style='color: #38bdf8; margin:0;'>📡 Telecom Site Audit AI Pipeline</h3>
    <p style='color: #94a3b8; margin:2px 0 0 0; font-size:12px;'>Dual Engine: Nemotron Parse 2.0 (PDF) + Gemini/DeepSeek (Vision Reasoning)</p>
</div>
""", unsafe_allow_html=True)

tab_audit, tab_pm, tab_combined = st.tabs([
    "📸 Field Photo Audit", 
    "📄 PM Checksheet Analyzer (Nemotron)", 
    "📊 Combined PDF + Photo Technical Report"
])

# ---------------------------------------------------------
# TAB 1: Field Audit (Photo Focus)
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

    if st.button("📤 Run Field Audit", width="stretch", disabled=(not manual_site_input or not is_location_valid)):
        if not uploaded_files:
            st.warning("Please attach at least one photo.")
        else:
            with st.spinner(f"⚡ Analyzing field photos with {ai_provider}..."):
                try:
                    pil_images = [optimize_image(f) for f in uploaded_files]

                    SYSTEM_PROMPT = (
                        f"You are a telecom audit engineer inspecting physical equipment, cabinet status, and NTG Asset Tagging.\n"
                        f"Analyzing Site ID: {manual_site_input} | Technician: {tech_name_input or 'Unassigned'}\n\n"
                        f"MANDATORY OUTPUT FORMAT:\n"
                        f"### 1. EQUIPMENT QUANTITY COUNT & AUDIT\n"
                        f"| Equipment / Asset Description | Identified Model / Brand | Quantities Detected | Photo Status |\n"
                        f"| :--- | :--- | :--- | :--- |\n\n"
                        f"### 2. NTG ASSET TAGGING VERIFICATION\n"
                        f"- Asset barcode visibility, port labeling, cable dressing.\n\n"
                        f"### 3. FINAL VERDICT & DEFECTS\n"
                        f"- **Final Verdict:** [PASS / PASS WITH CONCERNS / FAIL]\n"
                        f"- **Identified Defects:** Detail physical issues.\n"
                        f"- **Corrective Actions:** Required steps."
                    )

                    report_text = run_ai_audit_call(
                        provider=ai_provider,
                        prompt=SYSTEM_PROMPT,
                        text_content="",
                        pil_images=pil_images,
                        is_json=False
                    )

                    if report_text:
                        st.subheader("📋 Field Audit Report")
                        st.markdown(report_text)
                        
                        status_verdict = "FAIL" if "FAIL" in report_text.upper() else ("PASS WITH CONCERNS" if "CONCERNS" in report_text.upper() else "PASS")
                        save_report_to_supabase(manual_site_input, tech_name_input or 'Unassigned', status_verdict, report_text, user_lat, user_lon)

                except Exception as e:
                    st.error(f"Audit processing failed: {str(e)}")

# ---------------------------------------------------------
# TAB 2: Multi-PM Analyzer (Nemotron Parse Engine)
# ---------------------------------------------------------
with tab_pm:
    st.subheader("📄 Multi-PM Checksheet Analyzer (Powered by Nemotron Parse 2.0)")
    
    uploaded_pdfs = st.file_uploader(
        "Upload PM Checksheet PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="pm_pdf_uploader"
    )

    if uploaded_pdfs:
        st.info(f"📂 **{len(uploaded_pdfs)}** PDF file(s) ready for parsing.")

        if st.button("🚀 Parse PM PDFs with Nemotron", width="stretch"):
            all_site_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, pdf_file in enumerate(uploaded_pdfs):
                status_text.text(f"⚙️ Running Nemotron Parse 2.0 ({idx+1}/{len(uploaded_pdfs)}): {pdf_file.name}")
                
                _, resized_page_images = process_and_resize_pdf(pdf_file)

                try:
                    # Step 1: High-precision OCR page-by-page via Nemotron Parse 2.0
                    parsed_markdown = run_nemotron_pdf_parser(resized_page_images)

                    # Step 2: Convert parsed Markdown into structured JSON format
                    JSON_EXTRACT_PROMPT = """You are a JSON formatting assistant. Read the provided document text and convert it strictly into JSON format:
{
  "site_id": "Extracted Site ID",
  "vendor_technician": "Technician Name",
  "pm_date": "YYYY-MM-DD",
  "verdict": "APPROVED" | "APPROVED WITH CONCERNS" | "REJECTED",
  "missing_equipment_photos": ["Short note"],
  "critical_remarks": ["Short remark"],
  "supervisor_focus_notes": ["Short action"]
}"""

                    json_raw = run_ai_audit_call(
                        provider=ai_provider,
                        prompt=JSON_EXTRACT_PROMPT,
                        text_content=parsed_markdown,
                        pil_images=[],
                        is_json=True
                    )

                    parsed_json = parse_model_json(json_raw)
                    parsed_json["filename"] = pdf_file.name
                    all_site_data.append(parsed_json)

                except Exception as e:
                    all_site_data.append({
                        "filename": pdf_file.name,
                        "site_id": "PARSE_ERR",
                        "vendor_technician": "N/A",
                        "pm_date": "N/A",
                        "verdict": "REJECTED",
                        "missing_equipment_photos": ["Parse Failure"],
                        "critical_remarks": [f"Error: {str(e)}"],
                        "supervisor_focus_notes": ["Check PDF formatting"]
                    })

                progress_bar.progress((idx + 1) / len(uploaded_pdfs))

            status_text.success("✅ Nemotron PDF Parsing completed!")
            st.session_state["pm_analysis_results"] = all_site_data

    if "pm_analysis_results" in st.session_state and st.session_state["pm_analysis_results"]:
        results = st.session_state["pm_analysis_results"]
        
        summary_rows = [{
            "Site ID": r.get("site_id", "N/A"),
            "Technician": r.get("vendor_technician", "N/A"),
            "Date": r.get("pm_date", "N/A"),
            "Verdict": r.get("verdict", "N/A"),
            "Critical Remarks": " | ".join(r.get("critical_remarks", [])) if isinstance(r.get("critical_remarks"), list) else str(r.get("critical_remarks", "")),
            "Filename": r.get("filename", "")
        } for r in results]
        
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, width="stretch")

        st.download_button(
            label="📥 Download Summary (.xlsx)",
            data=convert_df_to_excel(df_summary, sheet_name='PM_Summary'),
            file_name="Multi_PM_Summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---------------------------------------------------------
# TAB 3: Combined Technical Report (PDFs + Field Photos)
# ---------------------------------------------------------
with tab_combined:
    st.subheader("📊 Combined Multi-Modal Technical Report")
    st.caption("Cross-references PDF checksheet data against uploaded field site photos.")

    col_pdf_in, col_img_in = st.columns(2)
    with col_pdf_in:
        comb_pdf = st.file_uploader("Upload Site PM PDF", type=["pdf"], key="comb_pdf")
    with col_img_in:
        comb_photos = st.file_uploader("Upload Physical Site Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="comb_photos")

    if st.button("⚡ Generate Combined Technical Report", width="stretch", disabled=(not comb_pdf or not comb_photos)):
        with st.spinner("Processing PDF via Nemotron & Analyzing Photos via Vision Engine..."):
            try:
                # 1. Parse PDF page-by-page with Nemotron Parse 2.0
                _, pdf_imgs = process_and_resize_pdf(comb_pdf)
                extracted_pdf_text = run_nemotron_pdf_parser(pdf_imgs)

                # 2. Prepare field photos
                pil_photos = [optimize_image(p) for p in comb_photos]

                # 3. Formulate Cross-Verification Prompt
                COMBINED_PROMPT = """You are a Lead Telecom Operations Supervisor compiling a comprehensive Technical Audit Report.

Cross-reference the EXTRACTED PDF CHECKSHEET TEXT against the ATTACHED FIELD SITE PHOTOS.

Produce a detailed report with the following structure:
1. EXECUTIVE SUMMARY & SITE METADATA (Site ID, Vendor Technician, PM Date)
2. CHECKSHEET DATA AUDIT (Summary of claims made in the PDF document)
3. PHYSICAL VISUAL AUDIT (Findings from physical photos: Cabinet status, Cable dressing, Rectifiers, Batteries, NTG Tags)
4. CROSS-VERIFICATION & DISCREPANCY ANALYSIS (Highlight differences between what technician claimed on PDF vs actual physical evidence in photos)
5. FINAL SUPERVISOR VERDICT & ACTION PLAN (PASS / PASS WITH CONCERNS / REJECTED + Corrective actions)"""

                combined_report = run_ai_audit_call(
                    provider=ai_provider,
                    prompt=COMBINED_PROMPT,
                    text_content=extracted_pdf_text,
                    pil_images=pil_photos,
                    is_json=False
                )

                st.markdown("---")
                st.markdown(combined_report)

            except Exception as e:
                st.error(f"Failed to generate combined report: {str(e)}")
