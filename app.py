import sys
import os
import io
import time
import re
import json
import difflib
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from google import genai
from google.genai import types

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
# Image Optimization Engine
# ---------------------------------------------------------
def optimize_image(uploaded_file, max_size=(2048, 2048), quality=95):
    """Optimizes field photos for high-resolution visual inspection."""
    img = Image.open(uploaded_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------------------
# PDF Processing Engine
# ---------------------------------------------------------
def process_and_resize_pdf(pdf_file, max_chars=25000, target_dpi=220, max_size=(2048, 2048), max_pages=15):
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
                img.save(buf, format="JPEG", quality=85, optimize=True)
                buf.seek(0)
                resized_images.append(Image.open(buf))

    except Exception as e:
        st.error(f"Error processing PDF '{pdf_file.name}': {str(e)}")

    return text_content, resized_images

# ---------------------------------------------------------
# Robust JSON Parser
# ---------------------------------------------------------
def parse_gemini_json(raw_text: str) -> Dict[str, Any]:
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
        if not repaired.endswith("}"):
            if not repaired.endswith('"') and not repaired.endswith(']'):
                repaired += '"'
            if not repaired.endswith("}"):
                repaired += "\n}"
        return json.loads(repaired)
    except Exception:
        raise ValueError(f"Failed to parse model output: {raw_text[:150]}...")

# ---------------------------------------------------------
# Resilient Dynamic Model Caller with Exponential Backoff
# ---------------------------------------------------------
def generate_gemini_content_robust(client, contents, config, max_retries=3):
    configured_model = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL")
    candidate_models = []
    if configured_model:
        candidate_models.append(configured_model)
    
    candidate_models.extend(["gemini-2.5-pro", "gemini-2.5-flash"])
    
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_error = None
    for model_name in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                return response
            except Exception as err:
                last_error = err
                err_msg = str(err).lower()

                if "404" in err_msg or "not_found" in err_msg:
                    break

                if any(code in err_msg for code in ["503", "unavailable", "500", "internal", "429", "resource_exhausted"]):
                    sleep_time = (attempt + 1) * 3
                    time.sleep(sleep_time)
                    continue
                else:
                    break

    raise last_error

# ---------------------------------------------------------
# FUZZY DEDUPLICATION ENGINE
# ---------------------------------------------------------
def is_similar_text(text1: str, text2: str, threshold: float = 0.85) -> bool:
    return difflib.SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio() >= threshold

def deduplicate_findings(findings: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    unique_findings = []

    for f in findings:
        is_duplicate = False
        f_asset = str(f.get("asset", "")).lower().strip()
        f_obs = str(f.get("observation", "")).lower().strip()

        for u in unique_findings:
            u_asset = str(u.get("asset", "")).lower().strip()
            u_obs = str(u.get("observation", "")).lower().strip()

            if f_asset == u_asset or is_similar_text(f_asset, u_asset, 0.80):
                if is_similar_text(f_obs, u_obs, similarity_threshold):
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_findings.append(f)

    return unique_findings

# ---------------------------------------------------------
# HIGH-PRECISION PIPELINE ENGINE (96%+ ACCURACY TARGET)
# ---------------------------------------------------------

# Stage 1: Photo Set Completeness Agent
def validate_photo_set(client, images: List[Image.Image]) -> Dict[str, bool]:
    validation_prompt = """
You are a Lead Telecom PM Auditor.
Check whether the submitted photos contain:
1. Tower close-up
2. Antenna view
3. Cabinet internal view
4. ATS internal view
5. Battery view
6. DG view
7. DG controller display
"""
    schema = {
        "type": "OBJECT",
        "properties": {
            "tower_closeup": {"type": "BOOLEAN"},
            "antenna_view": {"type": "BOOLEAN"},
            "cabinet_open": {"type": "BOOLEAN"},
            "ats_open": {"type": "BOOLEAN"},
            "battery_visible": {"type": "BOOLEAN"},
            "dg_visible": {"type": "BOOLEAN"},
            "dg_display": {"type": "BOOLEAN"}
        },
        "required": ["tower_closeup", "antenna_view", "cabinet_open", "ats_open", "battery_visible", "dg_visible", "dg_display"]
    }

    try:
        response = generate_gemini_content_robust(
            client=client,
            contents=[validation_prompt, *images],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema
            )
        )
        return parse_gemini_json(response.text)
    except Exception:
        return {
            "tower_closeup": False,
            "antenna_view": False,
            "cabinet_open": False,
            "ats_open": False,
            "battery_visible": False,
            "dg_visible": False,
            "dg_display": False
        }

# Stage 2: Precision Defect Detection Agent
def detect_defects(client, image: Image.Image, photo_id: int) -> List[Dict[str, Any]]:
    prompt = f"""You are a Senior Telecom Quality Assurance Inspector auditing photo #{photo_id}.

IMPORTANT RULES FOR HIGH ACCURACY:
1. Report ONLY visible, undeniable physical facts.
2. Do NOT guess or infer unshown components.
3. Strict Confidence Rules:
   - 100 = Clearly visible physical defect.
   - 90 = Visible but partially shadowed or distant.
   - 80 = Borderline or uncertain observation.
   - Below 80 = DO NOT REPORT.

4. Severity Guidelines:
   - Critical: Exposed live wires, water ingress risk, active hardware hazard.
   - Major: Missing protective cover (ATS/Trunking), unbundled feeder cables, missing ground lug.
   - Minor: Missing barcode label, slight paint scratch, minor cable alignment.
   - Observation: Compound housekeeping, minor dust inside cabinet.
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "findings": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "asset": {"type": "STRING"},
                        "category": {"type": "STRING"},
                        "observation": {"type": "STRING"},
                        "evidence": {"type": "STRING"},
                        "confidence": {"type": "INTEGER"},
                        "severity": {
                            "type": "STRING",
                            "enum": ["Critical", "Major", "Minor", "Observation"]
                        }
                    },
                    "required": ["asset", "category", "observation", "evidence", "confidence", "severity"]
                }
            }
        },
        "required": ["findings"]
    }

    try:
        response = generate_gemini_content_robust(
            client=client,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.1,
                response_mime_type="application/json",
                response_schema=schema
            )
        )
        parsed = parse_gemini_json(response.text)
        raw_findings = parsed.get("findings", [])
        
        for f in raw_findings:
            f["photo_id"] = photo_id

        return raw_findings
    except Exception:
        return []

# Stage 3: Targeted Verification Agent
def verify_defect(client, image: Image.Image, finding: Dict[str, Any]) -> bool:
    prompt = f"""
Verify the following telecom finding against the attached photo.

Asset: {finding.get('asset', '')}
Observation: {finding.get('observation', '')}
Evidence: {finding.get('evidence', '')}

Respond ONLY with:
VERIFIED
or
NOT VERIFIED
"""
    try:
        response = generate_gemini_content_robust(
            client=client,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.1
            )
        )
        answer = response.text.strip().upper()
        if answer.startswith("VERIFIED") and not answer.startswith("NOT VERIFIED"):
            return True
        return False
    except Exception:
        return False

# Stage 4: Supervisor Pass (Asset Inventory & Harmonization)
def supervisor_review(client, findings: List[Dict[str, Any]], photo_status: Dict[str, bool]) -> Dict[str, Any]:
    prompt = f"""You are a Lead Telecom QA Supervisor performing a final review of audit findings across all site photos.

Your Objectives:
1. Identify all physical telecom assets present on site across all photos (e.g., Tower, Outdoor Cabinet, ATS Panel, Battery Bank, Diesel Generator, Rectifier).
2. Clean up and harmonize findings: remove duplicate findings, merge equivalent observations across photos, and keep valid findings concise and clear.
3. Return the consolidated list of approved findings along with the complete asset inventory.

RAW FINDINGS:
{json.dumps(findings, indent=2)}

PHOTO COMPLETENESS:
{json.dumps(photo_status, indent=2)}
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "detected_assets": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "approved_findings": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "photo_id": {"type": "INTEGER"},
                        "asset": {"type": "STRING"},
                        "category": {"type": "STRING"},
                        "observation": {"type": "STRING"},
                        "evidence": {"type": "STRING"},
                        "confidence": {"type": "INTEGER"},
                        "severity": {
                            "type": "STRING",
                            "enum": ["Critical", "Major", "Minor", "Observation"]
                        }
                    },
                    "required": ["photo_id", "asset", "category", "observation", "evidence", "confidence", "severity"]
                }
            }
        },
        "required": ["detected_assets", "approved_findings"]
    }

    try:
        response = generate_gemini_content_robust(
            client=client,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=0.1,
                response_mime_type="application/json",
                response_schema=schema
            )
        )
        return parse_gemini_json(response.text)
    except Exception:
        return {
            "detected_assets": ["Telecom Infrastructure"],
            "approved_findings": findings
        }

# Orchestrator
def run_telecom_audit_pipeline(client, images: List[Image.Image]) -> Dict[str, Any]:
    photo_status = validate_photo_set(client, images)
    raw_candidates = []

    for idx, img in enumerate(images):
        photo_id = idx + 1
        detected = detect_defects(client, img, photo_id)

        for finding in detected:
            confidence = finding.get("confidence", 0)

            if confidence >= 90:
                raw_candidates.append(finding)
            elif confidence >= 80:
                if verify_defect(client, img, finding):
                    raw_candidates.append(finding)

    deduplicated = deduplicate_findings(raw_candidates, similarity_threshold=0.85)
    supervisor_result = supervisor_review(client, deduplicated, photo_status)

    return {
        "photos": photo_status,
        "assets": supervisor_result.get("detected_assets", []),
        "findings": supervisor_result.get("approved_findings", [])
    }

# Rule-Based Verdict Engine
def calculate_verdict(findings: List[Dict[str, Any]], photos: Dict[str, bool]) -> Dict[str, Any]:
    mandatory_photos = {
        "tower_closeup": "Tower Closeup",
        "cabinet_open": "Cabinet Interior"
    }
    optional_photos = {
        "antenna_view": "Antenna View",
        "ats_open": "ATS Interior",
        "battery_visible": "Battery Bank",
        "dg_visible": "Diesel Generator View",
        "dg_display": "DG Controller Display"
    }

    missing_mandatory = [name for key, name in mandatory_photos.items() if not photos.get(key)]
    missing_optional = [name for key, name in optional_photos.items() if not photos.get(key)]

    critical_count = sum(1 for f in findings if f.get("severity") == "Critical")
    major_count = sum(1 for f in findings if f.get("severity") == "Major")
    minor_count = sum(1 for f in findings if f.get("severity") == "Minor")

    verdict = "PASS"
    if critical_count > 0 or len(missing_mandatory) > 0:
        verdict = "REJECTED"
    elif major_count > 3:
        verdict = "REJECTED"
    elif major_count > 0 or minor_count > 2 or len(missing_optional) > 0:
        verdict = "PASS WITH CONCERNS"

    return {
        "verdict": verdict,
        "critical_count": critical_count,
        "major_count": major_count,
        "minor_count": minor_count,
        "missing_mandatory": missing_mandatory,
        "missing_optional": missing_optional
    }

# Excel Export Generator
def generate_excel_report(df: pd.DataFrame, site_id: str, verdict_data: Dict[str, Any]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Audit Findings", index=False)
        
        summary_df = pd.DataFrame([
            {"Metric": "Site ID", "Value": site_id},
            {"Metric": "Final Verdict", "Value": verdict_data["verdict"]},
            {"Metric": "Critical Defects", "Value": verdict_data["critical_count"]},
            {"Metric": "Major Defects", "Value": verdict_data["major_count"]},
            {"Metric": "Minor Defects", "Value": verdict_data["minor_count"]},
            {"Metric": "Missing Mandatory Photos", "Value": ", ".join(verdict_data["missing_mandatory"]) or "None"},
            {"Metric": "Missing Secondary Photos", "Value": ", ".join(verdict_data["missing_optional"]) or "None"}
        ])
        summary_df.to_excel(writer, sheet_name="Executive Summary", index=False)
        
    output.seek(0)
    return output.getvalue()

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
            gemini_key = st.secrets.get("GEMINI_API_KEY") or
