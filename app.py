import sys
import os
import io
import time
import re
import json
import smtplib
import base64
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
from openai import OpenAI

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
# Helper: Aggressive Image Optimization & Base64
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

def PIL_to_base64_data_url(pil_img, quality=75):
    """Converts PIL Image to Base64 Data URL for OpenAI/NVIDIA API."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

# ---------------------------------------------------------
# Helper: PDF Text & Resized Image Extractor
# ---------------------------------------------------------
def process_and_resize_pdf(pdf_file, max_chars=6000, target_dpi=100, max_size=(800, 800)):
    """Extracts text and converts PDF pages into downscaled images."""
    text_content = ""
    resized_images = []

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
            text_content = text_content[:max_chars] + "... [TRUNCATED FOR SPEED]"

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
# Helper: Robust JSON Output Parser
# ---------------------------------------------------------
def parse_model_json(raw_text):
    """Cleans markdown formatting and repairs common JSON truncation errors."""
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
# Helper: Unified AI Model Call Manager
# ---------------------------------------------------------
def run_ai_audit_call(provider, prompt, images, is_json=False):
    """Handles prompt + image execution across Gemini and NVIDIA API formats."""
    if provider == "Google Gemini":
        gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("Missing GEMINI_API_KEY!")
        
        client = genai.Client(api_key=gemini_key)
        configured_model = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
        
        gen_config = types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=0.0,
            response_mime_type="application/json" if is_json else "text/plain"
        )
        
        response = client.models.generate_content(
            model=configured_model,
            contents=[*images],
            config=gen_config
        )
        return response.text

    elif provider == "NVIDIA (DeepSeek)":
        nvidia_key = st.secrets.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not nvidia_key:
            raise ValueError("Missing NVIDIA_API_KEY!")
        
        client = OpenAI(
            base_url="[https://integrate.api.nvidia.com/v1](https://integrate.api.nvidia.com/v1)",
            api_key=nvidia_key
        )
        
        messages_payload = [{"type": "text", "text": prompt}]
        for img in images:
            if isinstance(img, Image.Image):
                b64_url = PIL_to_base64_data_url(img)
                messages_payload.append({"type": "image_url", "image_url": {"url": b64_url}})
            elif isinstance(img, str):
                messages_payload.append({"type": "text", "text": img})

        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": messages_payload}],
            temperature=0.1,
            max_tokens=2048
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
    page_title="Telecom Audit AI",
    page_icon="📡",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 1rem;
        padding-right
