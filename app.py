import sys
import os
import io
import time
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from math import radians, cos, sin, asin, sqrt
import resend
from google import genai
from google.genai.errors import APIError
from streamlit_js_eval import get_geolocation
from supabase import create_client, Client

# PyZBar for barcode/QR reading (fails gracefully if library is missing)
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

# ---------------------------------------------------------
# 0. Fix Import Paths & Absolute Workspace Directory
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from kml_parser import parse_telecom_kml
from geo_utils import find_nearby_sites

# ---------------------------------------------------------
# Helper: Extract Pure Site Code Only
# ---------------------------------------------------------
def get_clean_site_id(raw_str):
    if not raw_str or str(raw_str).strip() in ["-- Select Site --", "-- No Sites Found --"]:
        return ""
    return str(raw_str).strip().upper()

# ---------------------------------------------------------
# Helper: Barcode & Label Scanner
# ---------------------------------------------------------
def scan_equipment_barcodes(uploaded_files):
    if not PYZBAR_AVAILABLE:
        return []

    scanned_items = []
    for idx, file in enumerate(uploaded_files):
        try:
            file.seek(0)
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            file.seek(0)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                barcodes = pyzbar_decode(img)
                for barcode in barcodes:
                    scanned_items.append({
                        "Photo": f"Photo #{idx + 1}",
                        "Type": barcode.type,
                        "Barcode / Serial": barcode.data.decode("utf-8")
                    })
        except Exception:
            continue
    return scanned_items

# ---------------------------------------------------------
# Helper: Email Dispatcher via Gmail SMTP (Fallback to Resend)
# ---------------------------------------------------------
def send_email_notification(site_id, technician, status, report_text, user_lat, user_lon):
    receiver_email = st.secrets.get("ADMIN_RECEIVER_EMAIL") or os.environ.get("ADMIN_RECEIVER_EMAIL") or "mustafa.khalid@asiacell.com"
    sender_email = st.secrets.get("SENDER_EMAIL") or os.environ.get("SENDER_EMAIL") or "mustafa.khalid@asiacell.com"
    sender_password = st.secrets.get("SENDER_PASSWORD") or os.environ.get("SENDER_PASSWORD")
    
    gmail_server = st.secrets.get("GMAIL_SERVER") or os.environ.get("GMAIL_SERVER") or "smtp.gmail.com"
    gmail_port = int(st.secrets.get("GMAIL_PORT") or os.environ.get("GMAIL_PORT") or 587)

    status_color = "#16a34a" if status == "PASS" else ("#ca8a04" if "CONCERNS" in status else "#dc2626")

    html_body = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px;">
        <h2 style="color: #0284c7; margin-bottom: 5px;">📡 NGT Telecom Site Audit Report</h2>
        <hr style="border: 0; border-top: 1px solid #eee;">
        
        <table style="width: 100%; margin-top: 15px; font-size: 14px;">
            <tr><td><strong>Site ID:</strong></td><td>{site_id}</td></tr>
            <tr><td><strong>Technician:</strong></td><td>{technician}</td></tr>
            <tr><td><strong>Coordinates:</strong></td><td>{user_lat}, {user_lon}</td></tr>
            <tr><td><strong>Audit Status:</strong></td><td><span style="background-color: {status_color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{status}</span></td></tr>
        </table>

        <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">

        <h3 style="color: #333;">🤖 NGT Equipment Inventory & Inspection Findings</h3>
        <div style="background-color: #f8fafc; padding: 15px; border-left: 4px solid #0284c7; border-radius: 4px; white-space: pre-wrap; font-size: 13px; line-height: 1.6;">
{report_text}
        </div>

        <p style="font-size: 11px; color: #94a3b8; margin-top: 25px; text-align: center;">
            Automated Audit Notification • Asiacell R3-BAG-CLS5
        </p>
    </div>
    """

    if sender_password and sender_password != "your-actual-asiacell-password":
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🚨 Site Audit Report: {site_id} [{status}]"
            msg["From"] = sender_email
            msg["To"] = receiver_email
            msg.attach(MIMEText(html_body, "html"))

            server = smtplib.SMTP(gmail_server, gmail_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            return True
        except Exception as e:
            st.warning(f"⚠️ Gmail SMTP dispatch failed ({str(e)}). Attempting Resend API...")

    resend_key = st.secrets.get("RESEND_API_KEY") or os.environ.get("RESEND_API_KEY")
    if resend_key:
        try:
            resend.api_key = resend_key
            resend.Emails.send({
                "from": "Telecom Audit <onboarding@resend.dev>",
                "to": receiver_email,
                "subject": f"🚨 Site Audit Report: {site_id} [{status}]",
                "html": html_body
            })
            return True
        except Exception as e:
            st.warning(f"⚠️ Resend dispatch failed: {str(e)}")

    st.warning("⚠️ Email notification skipped: Configure Gmail App Password or Resend API key.")
    return False

# ---------------------------------------------------------
# Helper: Supabase Client Connection
# ---------------------------------------------------------
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)

def save_report_to_supabase(site_id, technician, status, report_text, user_lat, user_lon):
    try:
        supabase = get_supabase_client()
        if not supabase:
            st.error("❌ Supabase URL or Key missing in Streamlit Secrets!")
            return False

        data = {
            "site_id": site_id,
            "technician": technician,
            "coordinates": f"{user_lat}, {user_lon}" if user_lat else "N/A",
            "status": status,
            "report_text": report_text
        }
        supabase.table("audit_reports").insert(data).execute()
        send_email_notification(site_id, technician, status, report_text, user_lat, user_lon)
        return True
    except Exception as e:
        st.error(f"⚠️ Database submission failed: {str(e)}")
        return False

def fetch_supabase_reports():
    try:
        supabase = get_supabase_client()
        if not supabase:
            return pd.DataFrame()
        res = supabase.table("audit_reports").select("*").order("created_at", desc=True).execute()
        return pd.DataFrame(res.data)
    except Exception as e:
        st.error(f"⚠️ Failed to fetch remote reports: {str(e)}")
        return pd.DataFrame()

# ---------------------------------------------------------
# Helper: Convert DataFrame to Excel Binary Buffer
# ---------------------------------------------------------
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Audit_Reports')
    output.seek(0)
    return output.getvalue()

# ---------------------------------------------------------
# Helper: Haversine Distance Formula (km)
# ---------------------------------------------------------
def calculate_distance_km(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        return c * 6371.0
    except (ValueError, TypeError):
        return float('inf')

# ---------------------------------------------------------
# 1. Page Config & Custom Dark UI Styling
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

logged_user = st.session_state.get('logged_user')
st.sidebar.write(f"Logged in as: **{logged_user}**")
if st.sidebar.button("Log Out"):
    st.session_state["authenticated"] = False
    st.rerun()

# ---------------------------------------------------------
# 3. Absolute Path KML Resolver
# ---------------------------------------------------------
KML_EXACT_PATH = os.path.join(BASE_DIR, "data", "sites.kml")

@st.cache_data(ttl=60)
def load_kml_dataset():
    if os.path.exists(KML_EXACT_PATH):
        return parse_telecom_kml(KML_EXACT_PATH)
    
    data_dir = os.path.join(BASE_DIR, "data")
    if os.path.exists(data_dir):
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.kml')]
        if files:
            return parse_telecom_kml(files[0])
            
    return []

raw_sites = load_kml_dataset()
df_sites = pd.DataFrame(raw_sites)

# ---------------------------------------------------------
# 4. Header & Navigation Tabs
# ---------------------------------------------------------
st.markdown("""
    <div class='header-card'>
        <h2 style='color: #00d2ff; margin:0;'>📡 Telecom Site Audit AI (NGT Inventory)</h2>
        <p style='color: #8e9aaf; margin:4px 0 0 0;'>Designed by Mustafa Khalid / Supervisor / R3-BAG-CLS5</p>
    </div>
""", unsafe_allow_html=True)

if logged_user == "admin":
    tab_audit, tab_reports = st.tabs(["🔍 Field Site Audit & NGT Scanner", "📊 Admin Remote Reports Dashboard"])
else:
    tab_audit = st.container()
    tab_reports = None

# ---------------------------------------------------------
# TAB 1: Field Site Audit
# ---------------------------------------------------------
with tab_audit if logged_user == "admin" else st.container():
    col_site, col_tech = st.columns(2)

    with col_site:
        # Direct text entry for Site ID
        manual_site_input = st.text_input(
            "ENTER SITE ID", 
            placeholder="e.g. BAG0123"
        ).strip().upper()
        
        selected_site_code = manual_site_input if manual_site_input else ""

    with col_tech:
        tech_name_input = st.text_input("TECHNICIAN NAME", placeholder="e.g. Alaa Fadel").strip()

    loc = get_geolocation()
    user_lat, user_lon = None, None

    if loc and 'coords' in loc:
        user_lat = loc['coords']['latitude']
        user_lon = loc['coords']['longitude']
        st.sidebar.success(f"🌐 GPS Active: {user_lat:.4f}, {user_lon:.4f}")
    else:
        st.sidebar.warning("⚠️ GPS inactive. Please enable browser location permissions.")

    is_location_valid = False
    site_data = None

    if selected_site_code and not df_sites.empty:
        possible_cols = ['site_code', 'site_id', 'name', 'Site_Code', 'SiteID', 'Name']
        target_col = next((c for c in possible_cols if c in df_sites.columns), None)

        if target_col:
            matched = df_sites[df_sites[target_col].astype(str).str.strip().str.upper() == selected_site_code]
            
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
                            st.error(f"❌ Location Mismatch: You are **{distance:.2f} km** away from this site. Must be within **3 km**.")
                    else:
                        st.warning("⚠️ GPS Signal Required: Please enable device location permissions.")
            else:
                st.warning(f"⚠️ Site ID **{selected_site_code}** not found in the loaded KML dataset. You can still proceed if GPS validation is bypassed or verified.")
                # Allow submission if user manually confirms
                is_location_valid = True
        else:
            is_location_valid = True
    elif selected_site_code:
        # If KML dataset is empty or not loaded, allow field technician entry
        is_location_valid = True

    st.markdown("### PHOTOS")

    if "captured_photos" not in st.session_state:
        st.session_state["captured_photos"] = []

    uploaded_files = []

    if not selected_site_code:
        st.error("🔒 Photo upload and submission are locked. Enter a Site ID to begin.")
    elif not is_location_valid:
        st.error("🔒 Location verification failed. Ensure you are within 3 km of the site.")
    else:
        input_mode = st.radio("Choose Input Method:", ["Camera", "Gallery"], horizontal=True)

        if input_mode == "Camera":
            img_file = st.camera_input("Capture Site Photo")

            if img_file is not None:
                img_bytes = img_file.getvalue()
                if not any(p.getvalue() == img_bytes for p in st.session_state["captured_photos"]):
                    st.session_state["captured_photos"].append(img_file)

            col_clear, col_count = st.columns([1, 4])
            with col_clear:
                if st.button("🗑️ Clear Photos"):
                    st.session_state["captured_photos"] = []
                    st.rerun()

            uploaded_files = st.session_state["captured_photos"]

        else:
            img_files = st.file_uploader("Pick from gallery", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            if img_files:
                uploaded_files.extend(img_files)

        if uploaded_files:
            st.write(f"Selected Photos ({len(uploaded_files)}):")
            cols = st.columns(6)
            for idx, file in enumerate(uploaded_files):
                with cols[idx % 6]:
                    st.image(file, width=120)

    st.write("---")
    if st.button("📤 Submit Site Audit & NGT Report", use_container_width=True, disabled=(not selected_site_code or not is_location_valid)):
        if not uploaded_files:
            st.warning("Please capture or upload at least one site photo.")
        else:
            gemini_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

            if not gemini_key:
                st.error("❌ Missing Gemini API Key! Configure `GEMINI_API_KEY` in Streamlit Secrets.")
            else:
                with st.spinner("🏷️ Scanning Barcodes & Processing NGT Inventory with AI..."):
                    try:
                        barcodes_found = scan_equipment_barcodes(uploaded_files)
                        if barcodes_found:
                            st.markdown("#### 🏷️ Scanned Barcodes & Asset Labels")
                            st.dataframe(pd.DataFrame(barcodes_found), use_container_width=True)
                            barcode_summary = "\n".join([f"- [{b['Photo']}] ({b['Type']}) Serial: {b['Barcode / Serial']}" for b in barcodes_found])
                        else:
                            barcode_summary = "No machine-readable 1D/2D barcodes extracted by CV. Read human-printed labels directly from the photos."

                        client = genai.Client(api_key=gemini_key)
                        pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]

                        prompt = f"""
You are an expert telecommunications site audit engineer performing an NGT Equipment Asset & Quantity Inventory inspection.
Site ID: {selected_site_code}
Technician: {tech_name_input or 'Unassigned'}

Auto-Scanned Barcodes/Asset Labels:
{barcode_summary}

Analyze the provided image(s) thoroughly and generate a structured NGT site inspection and equipment count report:

1. **NGT EQUIPMENT QUANTITY COUNT & AUDIT**:
   - **Antennas**: Count RF sector antennas and microwave transmission dishes (note brand/type if visible).
   - **Batteries**: Count total lithium battery packs and lead-acid battery strings.
   - **Power Systems**: Count active Huawei ETP48/rectifier cabinets and rectifier modules.
   - **RAN / Transmission Equipment**: Count active RRUs/RRHs, BBU units, and RTN microwave ODUs.

2. **BARCODE & LABEL VERIFICATION**:
   - Cross-reference visible barcode tags and printed labels against the detected equipment.
   - List any unreadable, damaged, or missing asset barcode labels.

3. **INSTALLATION QUALITY & CABLING**:
   - Cable routing neatness, grounding connections, physical integrity, cleanliness.

4. **FINAL VERDICT & DEFECTS**:
   - PASS, PASS WITH CONCERNS, or FAIL (Include justification and required corrective actions).
                        """

                        target_model = st.secrets.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

                        max_retries = 3
                        report_text = None

                        for attempt in range(max_retries):
                            try:
                                response = client.models.generate_content(
                                    model=target_model,
                                    contents=[prompt, *pil_images]
                                )
                                report_text = response.text
                                break
                            except APIError as api_err:
                                if "429" in str(api_err) or "RESOURCE_EXHAUSTED" in str(api_err):
                                    if attempt < max_retries - 1:
                                        wait_sec = 15 * (attempt + 1)
                                        st.warning(f"⏳ Rate limit reached. Retrying in {wait_sec} seconds (Attempt {attempt + 1}/{max_retries})...")
                                        time.sleep(wait_sec)
                                    else:
                                        raise api_err
                                else:
                                    raise api_err

                        if report_text:
                            status_verdict = "PASS"
                            if "FAIL" in report_text.upper():
                                status_verdict = "FAIL"
                            elif "CONCERNS" in report_text.upper():
                                status_verdict = "PASS WITH CONCERNS"

                            st.subheader("📋 NGT Audit & Quantity Inventory Report")
                            st.markdown(report_text)

                            if save_report_to_supabase(selected_site_code, tech_name_input or 'Unassigned', status_verdict, report_text, user_lat, user_lon):
                                st.success(f"✅ NGT Audit report for Site **{selected_site_code}** successfully submitted to supervisor!")
                                st.session_state["captured_photos"] = []

                    except APIError as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            st.error("⏳ **API Quota Exceeded**: You've reached the daily rate limit. Switch to pay-as-you-go or `gemini-2.5-flash` in Secrets.")
                        else:
                            st.error(f"⚠️ Gemini API Error: {str(e)}")
                    except Exception as e:
                        st.error(f"⚠️ Audit submission failed: {str(e)}")

# ---------------------------------------------------------
# TAB 2: Admin Remote Dashboard
# ---------------------------------------------------------
if logged_user == "admin" and tab_reports is not None:
    with tab_reports:
        st.subheader("📊 Remote Site Audit Log (Supabase Database)")
        
        if st.button("🔄 Refresh Data"):
            st.rerun()

        df_reports = fetch_supabase_reports()

        if not df_reports.empty:
            st.dataframe(df_reports[['created_at', 'site_id', 'technician', 'coordinates', 'status', 'report_text']], use_container_width=True)
            
            excel_data = convert_df_to_excel(df_reports[['created_at', 'site_id', 'technician', 'coordinates', 'status', 'report_text']])
            
            st.download_button(
                label="📊 Download Audit History (Excel .xlsx)",
                data=excel_data,
                file_name="site_audit_reports.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("No remote records found in the database yet.")
