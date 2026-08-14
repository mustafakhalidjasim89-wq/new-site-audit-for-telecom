import os
import sys

# Ensure Vercel can locate imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from supabase import create_client, Client

# Page layout configuration
st.set_page_config(
    page_title="Telecom Site Audit Portal",
    page_icon="📡",
    layout="centered"
)

# Fetch Supabase Credentials
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL") or "https://dxtkctltwnghsfljjjym.supabase.co"
SUPABASE_KEY = os.getenv("VITE_SUPABASE_ANON_KEY") or ""

@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    supabase = get_supabase_client()
except Exception as e:
    st.error(f"Error connecting to Supabase: {e}")
    st.stop()

# Session State Initialization
if "session" not in st.session_state:
    st.session_state.session = None

# --- LOGIN SCREEN ---
if not st.session_state.session:
    st.title("📡 Telecom Site Audit Portal")
    st.subheader("Engineer Authentication")

    with st.form("login_form"):
        email = st.text_input("Email Address", placeholder="engineer@domain.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submit_button = st.form_submit_button("Sign In")

        if submit_button:
            if not email or not password:
                st.warning("Please enter both email and password.")
            else:
                with st.spinner("Authenticating..."):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.session_state.session = res.session
                        st.success("Login Successful!")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Authentication Failed: {err}")

# --- DASHBOARD ---
else:
    user_email = st.session_state.session.user.email
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📡 Site Audit Dashboard")
        st.caption(f"Logged in as: **{user_email}**")
    with col2:
        if st.button("Sign Out"):
            supabase.auth.sign_out()
            st.session_state.session = None
            st.rerun()

    st.divider()

    st.subheader("Site Audit Submission")
    with st.form("audit_submission"):
        site_id = st.text_input("Site ID / Name", placeholder="e.g., BGW_0123")
        tech = st.selectbox("Technology Generation", ["2G", "3G", "4G", "5G", "Multi-band"])
        power_status = st.selectbox("Power System (-48V DC)", ["Normal", "Battery Warning", "Rectifier Alarm", "Mains Failure"])
        notes = st.text_area("Audit Notes / Findings")
        
        submitted = st.form_submit_button("Submit Audit Record")
        if submitted:
            st.success(f"Audit record for {site_id} successfully submitted!")