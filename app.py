import sys
import os
import io
import time
import re
import json

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from math import radians, cos, sin, asin, sqrt

from google import genai
from google.genai import types

from streamlit_js_eval import get_geolocation
from supabase import create_client, Client
from pypdf import PdfReader

# PyMuPDF (fitz) for rendering PDF pages as images
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# ---------------------------------------------------------
# Path Resolution & Setup
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
# Image Optimization
# ---------------------------------------------------------
def optimize_image(uploaded_file, max_size=(1024, 1024), quality=80):
    img = Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer)

# ---------------------------------------------------------
# PDF Processing Engine
# ---------------------------------------------------------
def process_and_resize_pdf(pdf_file, max_chars=25000, target_dpi=120, max_size=(1024, 1024), max_pages=15):
    text_content = ""
    resized_images = []

    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)

        reader = PdfReader(io.BytesIO(pdf_bytes))
        raw_text = ""
        for page_num, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                raw_text += f"\n--- Page {page_num + 1} ---\n" + t

        text_content = re.sub(r'[ \t]+', ' ', raw_text).strip()
        if len(text_content) > max_chars:
            text_content = text_content[:max_chars] + "\n... [TRUNCATED]"

        if FITZ_AVAILABLE:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_idx, page in enumerate(doc):
                if page_idx >= max_pages:
                    break
                pix = page.get_pixmap(dpi=target_dpi)
                img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75, optimize=True)
                buf.seek(0)
                resized_images.append(Image.open(buf))

    except Exception as e:
        st.error(f"Error processing PDF '{pdf_file.name}': {str(e)}")

    return text_content, resized_images

# ---------------------------------------------------------
# Robust JSON Parser
# ---------------------------------------------------------
def parse_gemini_json(raw_text):
    if not raw_text:
        raise ValueError("Empty response received.")
    
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
        if not repaired.endswith("}"):
            if not repaired.endswith('"') and not repaired.endswith(']'):
                repaired += '"'
            if not repaired.endswith("}"):
                repaired += "\n}"
        return json.loads(repaired)
    except Exception:
        raise ValueError(f"Failed to parse model output: {raw_text[:150]}...")

# ---------------------------------------------------------
# Dynamic Model Caller
# ---------------------------------------------------------
def generate_gemini_content_robust(client, contents, config):
    configured_model = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL")
    candidate_models = []
    if configured_model:
        candidate_models.append(configured_model)
    candidate_models.extend(["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.5-flash"])
    
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
            if "404" in err_msg or "not_found" in err_msg:
                continue
            elif "429" in err_msg or "resource_exhausted" in err_msg:
                time.sleep(3)

    raise last_error

# ---------------------------------------------------------
# UI Layout & Tab Logic
# ---------------------------------------------------------
st.set_page_config(page_title="Telecom Site Audit AI", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .stApp { background-color: #121417; color: #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

tab_audit, tab_pm = st.tabs(["🔍 Field Audit & NTG", "📄 Multi-PM Analyzer"])

# ---------------------------------------------------------
# TAB 1: ACCURATE FIELD AUDIT
# ---------------------------------------------------------
with tab_audit:
    st.subheader("📡 High-Precision Field Quality Audit")
    
    col_site, col_tech = st.columns(2)
    with col_site:
        manual_site_input = st.text_input("SITE ID", placeholder="e.g., BAG0123").strip().upper()
    with col_tech:
        tech_name_input = st.text_input("TECHNICIAN / SUBCONTRACTOR", placeholder="e.g., Subcontractor Name").strip()

    uploaded_files = st.file_uploader("Upload Field Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("📤 Run Precision Field Audit", use_container_width=True, disabled=not manual_site_input):
        if not uploaded_files:
            st.warning("Please attach site photos to evaluate.")
        else:
            gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                st.error("Missing GEMINI_API_KEY!")
            else:
                with st.spinner("🔍 Performing granular visual analysis..."):
                    client = genai.Client(api_key=gemini_key)
                    pil_images = [optimize_image(f) for f in uploaded_files]

                    FIELD_SYSTEM_PROMPT = """
You are a Lead Senior Telecom Site Quality Inspector.

MISSION:
Perform a strict telecom infrastructure audit based ONLY on visible evidence from uploaded photos and documents.

ZERO HALLUCINATION RULE:
- Never assume hidden defects.
- Never mention equipment that is not visible.
- Never create findings without visual evidence.
- If evidence is insufficient, report: "Unable to verify from provided photos."

PHOTO QUALITY RULE:
If photos are blurry, distant, partially visible, or incorrectly captured, report this under Documentation & Inspection Quality.

DO NOT repeat findings across sections.

REQUIRED REPORT OUTPUT FORMAT:

### ⚙️ 1. EQUIPMENT INVENTORY & PHOTO VALIDATION
| Asset / Equipment | Detected Status | Photo Quality & Elevation | Compliance Finding |
| :--- | :--- | :--- | :--- |

### 🛠️ 2. GRANULAR DEFECT BREAKDOWN (HUMAN-GRADE AUDIT)
* **Tower & Antenna Assets:** [Specific defects or "No defects identified"]
* **Power Cabinet & ATS Dressing:** [Specific defects or "No defects identified"]
* **Cable Containment & Trays:** [Specific defects or "No defects identified"]
* **Site Housekeeping & Safety:** [Specific defects or "No defects identified"]

### 🎯 3. REQUIRED REWORK & ACTIONABLE DIRECTIVES
1. [Precise remediation action required from technician]
2. [Precise remediation action required from technician]

### 📊 4. FINAL ACCEPTANCE VERDICT
**VERDICT:** [PASS / PASS WITH CONCERNS / REJECTED]
**SUMMARY REASONING:** [One concise paragraph detailing exact grounds for verdict]
"""

                    report_text = generate_gemini_content_robust(
                        client=client,
                        contents=[f"SITE CODE: {manual_site_input}\nTECHNICIAN: {tech_name_input or 'Unassigned'}", *pil_images],
                        config=types.GenerateContentConfig(system_instruction=FIELD_SYSTEM_PROMPT, temperature=0.0)
                    )

                    st.markdown("---")
                    st.markdown(report_text)

# ---------------------------------------------------------
# TAB 2: MULTI-PM ANALYZER (STRICT ACCURACY)
# ---------------------------------------------------------
with tab_pm:
    st.subheader("📄 Multi-PM Checksheet Detailed Analyzer")
    
    uploaded_pdfs = st.file_uploader("Upload PM PDF Files", type=["pdf"], accept_multiple_files=True)

    if uploaded_pdfs and st.button("🚀 Analyze All PM Checksheets", use_container_width=True):
        gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            st.error("Missing GEMINI_API_KEY!")
        else:
            client = genai.Client(api_key=gemini_key)
            all_site_data = []

            STRICT_PM_SYSTEM_PROMPT = """
You are a Senior Telecom Operations Supervisor auditing submitted Preventive Maintenance (PM) checksheets.
Examine the provided text AND page images of the PM checksheets to identify specific, unique site non-conformities.

RULES TO PREVENT REPETITION AND INSURE ACCURACY:
- DO NOT duplicate findings across array fields.
- Use explicit, human-readable observation descriptions.
- Check explicitly for required photos: Tower Close-ups, Cabinet Open Doors, DG Overall, DG Display Screen (Running Hours).

EXTRACT AND RETURN ONLY JSON MATCHING THIS EXACT SCHEMA:
{
  "site_id": "Exact Site Code extracted from document",
  "vendor_technician": "Technician or Subcontractor Name",
  "pm_date": "YYYY-MM-DD",
  "verdict": "APPROVED" | "APPROVED WITH CONCERNS" | "REJECTED",
  "missing_equipment_photos": [
    "Specific missing mandatory photos (e.g., 'DG Controller screen image missing', 'Cabinet interior with doors open missing')"
  ],
  "critical_remarks": [
    "Specific physical findings (e.g., 'Dry grass around generator and fuel tank', 'ATS PVC tray missing top cover', 'Feeder cables unbundled on upper tower section')"
  ],
  "supervisor_focus_notes": [
    "Direct action required (e.g., 'Technician must return to clear vegetation within 3m buffer and upload close-up DG display photo')"
  ]
}
"""

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, pdf_file in enumerate(uploaded_pdfs):
                status_text.text(f"⚙️ Extracting data from ({idx+1}/{len(uploaded_pdfs)}): {pdf_file.name}")
                text_content, resized_page_images = process_and_resize_pdf(pdf_file, max_chars=25000, max_pages=15)
                
                payload = [f"FILENAME: {pdf_file.name}\nEXTRACTED CHECKLIST TEXT:\n{text_content}"]
                if resized_page_images:
                    payload.extend(resized_page_images)

                gen_config = types.GenerateContentConfig(
                    system_instruction=STRICT_PM_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=4096,
                    response_mime_type="application/json"
                )

                try:
                    raw_response = generate_gemini_content_robust(client=client, contents=payload, config=gen_config)
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
                        "missing_equipment_photos": ["Failed to extract PDF"],
                        "critical_remarks": [f"Parsing failure: {str(e)}"],
                        "supervisor_focus_notes": ["Verify file integrity"]
                    })

                progress_bar.progress((idx + 1) / len(uploaded_pdfs))

            status_text.success("Analysis Complete!")
            st.session_state["pm_analysis_results"] = all_site_data

    # Display Breakdown
    if "pm_analysis_results" in st.session_state and st.session_state["pm_analysis_results"]:
        results = st.session_state["pm_analysis_results"]
        st.markdown("### 📋 Multi-PM Summary Table")
        
        summary_rows = []
        for r in results:
            summary_rows.append({
                "Site ID": r.get("site_id", "N/A"),
                "Technician": r.get("vendor_technician", "N/A"),
                "Verdict": r.get("verdict", "N/A"),
                "Missing Photos": ", ".join(r.get("missing_equipment_photos", [])),
                "Critical Remarks": ", ".join(r.get("critical_remarks", [])),
                "Action Items": ", ".join(r.get("supervisor_focus_notes", [])),
                "File": r.get("filename", "")
            })
        
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
