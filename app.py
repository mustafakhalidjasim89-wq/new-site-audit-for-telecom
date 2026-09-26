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
# PDF Processing Engine (220 DPI & 2048x2048 Resolution)
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
# Resilient Model Caller with Retries (503 / 500 / 429)
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

                # Handle 404 immediately by switching model
                if "404" in err_msg or "not_found" in err_msg:
                    break  # Try next model

                # Retry on transient server, rate-limit, or overload errors
                if any(code in err_msg for code in ["503", "unavailable", "500", "internal", "429", "resource_exhausted"]):
                    sleep_time = (attempt + 1) * 3  # Exponential delay: 3s, 6s, 9s
                    time.sleep(sleep_time)
                    continue
                else:
                    break

    raise last_error

# ---------------------------------------------------------
# FUZZY DEDUPLICATION ENGINE
# ---------------------------------------------------------
def is_similar_text(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Calculates string similarity ratio using difflib SequenceMatcher."""
    return difflib.SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio() >= threshold

def deduplicate_findings(findings: List[Dict[str, Any]], similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Deduplicates findings based on asset match + fuzzy observation similarity."""
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
# MULTI-AGENT TELECOM AUDIT PIPELINE ENGINE
# ---------------------------------------------------------

# Stage 1: Photo Completeness Agent
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

# Stage 2: Defect Detection Agent (Single High-Precision Pass)
def detect_defects(client, image: Image.Image, photo_id: int) -> List[Dict[str, Any]]:
    prompt = f"""You are a Senior Telecom Quality Assurance Inspector auditing photo #{photo_id}.

IMPORTANT RULES:
1. Report ONLY visible, observable physical facts.
2. Never guess or infer unshown items.
3. Ignore anything not clearly visible in this specific photo.
4. Confidence Scoring Rules:
   - 100 = Clearly visible, undeniable physical issue.
   - 90 = Visible but partially obscured or shadowed.
   - 80 = Uncertain or borderline visible.
   - Any finding below 80 MUST NOT BE REPORTED.

5. Severity Guidelines:
   - Critical: Direct safety hazard, exposed live electrical wire, water ingress risk, active equipment malfunction.
   - Major: Missing protective cover (ATS/Trunking), unbundled/sagging cables, missing grounding lugs.
   - Minor: Missing asset barcode label, minor paint scratch, slight cable alignment issue.
   - Observation: Housekeeping item, minor dust/debris inside compound.
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

# Stage 3: Defect Verification Agent (Strict Prefix Checking)
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

# Stage 4: Global QA Supervisor Review Agent (Asset Inventory + Context Harmonization)
def supervisor_review(client, findings: List[Dict[str, Any]], photo_status: Dict[str, bool]) -> Dict[str, Any]:
    prompt = f"""You are a Lead Telecom QA Supervisor performing a final review of audit findings across all site photos.

Your Objectives:
1. Identify all physical telecom assets present on site across all photos (e.g., Tower, Outdoor Cabinet, ATS Panel, Battery Bank, Diesel Generator, Rectifier).
2. Clean up and harmonize findings: remove duplicate findings, merge equivalent observations across photos, and keep valid findings concise and clear.
3. Return the consolidated list of approved findings along with the complete asset inventory.

RAW FINDINGS SUBMITTED FOR REVIEW:
{json.dumps(findings, indent=2)}

PHOTO COMPLETENESS SUMMARY:
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
        # Fallback to unreviewed findings if supervisor call fails
        return {
            "detected_assets": ["Telecom Site Infrastructure"],
            "approved_findings": findings
        }

# Stage 5: Pipeline Orchestrator with Smart Confidence Gating
def run_telecom_audit_pipeline(client, images: List[Image.Image]) -> Dict[str, Any]:
    # 1. Check photo set completeness
    photo_status = validate_photo_set(client, images)
    raw_candidates = []

    # 2. Process each image with smart Python confidence rules
    for idx, img in enumerate(images):
        photo_id = idx + 1
        detected = detect_defects(client, img, photo_id)

        for finding in detected:
            confidence = finding.get("confidence", 0)

            # Rule 1: Auto-accept high-confidence detections (bypasses 2nd Gemini call)
            if confidence >= 90:
                raw_candidates.append(finding)
            # Rule 2: Verify borderline detections (80 <= confidence < 90)
            elif confidence >= 80:
                if verify_defect(client, img, finding):
                    raw_candidates.append(finding)
            # Rule 3: Reject < 80 automatically

    # 3. Apply Python fuzzy deduplication (0.85 threshold)
    deduplicated = deduplicate_findings(raw_candidates, similarity_threshold=0.85)

    # 4. Final Global Supervisor Review Pass (Asset Inventory + Consolidation)
    supervisor_result = supervisor_review(client, deduplicated, photo_status)

    return {
        "photos": photo_status,
        "assets": supervisor_result.get("detected_assets", []),
        "findings": supervisor_result.get("approved_findings", [])
    }

# Stage 6: Rule-Based Industry Verdict Calculation (Mandatory Photo Logic)
def calculate_verdict(findings: List[Dict[str, Any]], photos: Dict[str, bool]) -> Dict[str, Any]:
    # Define strictly mandatory photos required for site acceptance
    mandatory_photos = {
        "tower_closeup": "Tower Closeup",
        "cabinet_open": "Cabinet Interior"
    }
    
    # Optional / Secondary photos
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

# ---------------------------------------------------------
# EXCEL GENERATOR HELPER
# ---------------------------------------------------------
def generate_excel_report(df: pd.DataFrame, site_id: str, verdict_data: Dict[str, Any]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Sheet 1: Findings
        df.to_excel(writer, sheet_name="Audit Findings", index=False)
        
        # Sheet 2: Executive Summary
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
# STREAMLIT UI LAYOUT
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
# TAB 1: PRODUCTION FIELD AUDIT
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
                with st.spinner("🔍 Executing Audit Pipeline (Gated Detection, Deduplication & Supervisor Pass)..."):
                    client = genai.Client(api_key=gemini_key)
                    pil_images = [optimize_image(f) for f in uploaded_files]

                    # Run production multi-agent pipeline
                    audit_result = run_telecom_audit_pipeline(client, pil_images)
                    verdict_data = calculate_verdict(audit_result["findings"], audit_result["photos"])
                    verdict = verdict_data["verdict"]

                    st.markdown("---")
                    st.markdown("## Telecom Audit Report")
                    st.write(f"### Site: {manual_site_input} | Subcontractor: {tech_name_input or 'N/A'}")
                    
                    # Verdict banner
                    if verdict == "PASS":
                        st.success(f"### Verdict: {verdict}")
                    elif verdict == "PASS WITH CONCERNS":
                        st.warning(f"### Verdict: {verdict}")
                    else:
                        st.error(f"### Verdict: {verdict}")

                    # Verdict Breakdown Summary
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Critical Defects", verdict_data["critical_count"])
                    c2.metric("Major Defects", verdict_data["major_count"])
                    c3.metric("Minor Defects", verdict_data["minor_count"])
                    c4.metric("Missing Mandatory Photos", len(verdict_data["missing_mandatory"]))

                    # Asset Inventory Section
                    st.markdown("### Detected Site Asset Inventory")
                    st.info(", ".join(audit_result["assets"]) if audit_result["assets"] else "No major assets categorized.")

                    # Photo Completeness Section
                    st.markdown("### Photo Set Completeness Check")
                    photo_df = pd.DataFrame([audit_result["photos"]]).T.reset_index()
                    photo_df.columns = ["Photo View Requirement", "Submitted & Identified"]
                    st.dataframe(photo_df, use_container_width=True)

                    # Verified Findings Section
                    st.write("### Verified Findings")
                    if len(audit_result["findings"]) == 0:
                        st.success("No verified defects identified.")
                        findings_df = pd.DataFrame(columns=["photo_id", "severity", "asset", "category", "observation", "evidence", "confidence"])
                    else:
                        findings_df = pd.DataFrame(audit_result["findings"])
                        cols = ["photo_id", "severity", "asset", "category", "observation", "evidence", "confidence"]
                        findings_df = findings_df[[c for c in cols if c in findings_df.columns]]
                        st.dataframe(findings_df, use_container_width=True)

                    # Download Excel Report Button
                    excel_data = generate_excel_report(findings_df, manual_site_input, verdict_data)
                    st.download_button(
                        label="📥 Download Excel Audit Report",
                        data=excel_data,
                        file_name=f"{manual_site_input}_Audit_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

# ---------------------------------------------------------
# TAB 2: MULTI-PM ANALYZER
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

RULES TO PREVENT REPETITION AND ENSURE ACCURACY:
- Extract the exact Site ID and assign a confidence score (0-100) based on visual clarity.
- DO NOT duplicate findings across array fields.
- Use explicit, human-readable observation descriptions.
"""
            pm_schema = {
                "type": "OBJECT",
                "properties": {
                    "site_id": {"type": "STRING"},
                    "site_id_confidence": {"type": "INTEGER"},
                    "vendor_technician": {"type": "STRING"},
                    "pm_date": {"type": "STRING"},
                    "verdict": {
                        "type": "STRING",
                        "enum": ["APPROVED", "APPROVED WITH CONCERNS", "REJECTED"]
                    },
                    "missing_equipment_photos": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "critical_remarks": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    },
                    "supervisor_focus_notes": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    }
                },
                "required": ["site_id", "site_id_confidence", "vendor_technician", "pm_date", "verdict", "missing_equipment_photos", "critical_remarks", "supervisor_focus_notes"]
            }

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
                    top_p=0.1,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_schema=pm_schema
                )

                try:
                    raw_response = generate_gemini_content_robust(client=client, contents=payload, config=gen_config)
                    parsed = parse_gemini_json(raw_response.text)
                    parsed["filename"] = pdf_file.name
                    all_site_data.append(parsed)
                except Exception as e:
                    all_site_data.append({
                        "filename": pdf_file.name,
                        "site_id": "ERROR",
                        "site_id_confidence": 0,
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

    # Display Summary Table
    if "pm_analysis_results" in st.session_state and st.session_state["pm_analysis_results"]:
        results = st.session_state["pm_analysis_results"]
        st.markdown("### 📋 Multi-PM Summary Table")
        
        summary_rows = []
        for r in results:
            summary_rows.append({
                "Site ID": r.get("site_id", "N/A"),
                "ID Confidence (%)": r.get("site_id_confidence", "N/A"),
                "Technician": r.get("vendor_technician", "N/A"),
                "Verdict": r.get("verdict", "N/A"),
                "Missing Photos": ", ".join(r.get("missing_equipment_photos", [])),
                "Critical Remarks": ", ".join(r.get("critical_remarks", [])),
                "Action Items": ", ".join(r.get("supervisor_focus_notes", [])),
                "File": r.get("filename", "")
            })
        
        pm_df = pd.DataFrame(summary_rows)
        st.dataframe(pm_df, use_container_width=True)

        # Download PM Summary Excel
        pm_excel_buffer = io.BytesIO()
        with pd.ExcelWriter(pm_excel_buffer, engine="openpyxl") as writer:
            pm_df.to_excel(writer, sheet_name="PM Checksheet Summary", index=False)
        
        st.download_button(
            label="📥 Download PM Summary Excel",
            data=pm_excel_buffer.getvalue(),
            file_name="Multi_PM_Audit_Summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
