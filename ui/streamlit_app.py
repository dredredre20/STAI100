import io
import json
import tempfile
from pathlib import Path

import requests
import streamlit as st

API_BASE_URL = st.session_state.get("api_base_url", "http://127.0.0.1:8000") if "api_base_url" in st.session_state else "http://127.0.0.1:8000"

st.set_page_config(page_title="STAI100 Resume Intake", page_icon="📄", layout="wide")
st.title("STAI100 Resume Intake")
st.caption("Upload a resume PDF and inspect the parsed profile from the local AI pipeline.")

with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input("FastAPI base URL", value=API_BASE_URL)
    target_role = st.selectbox(
        "Target role",
        ["", "data_scientist", "cloud_engineering"],
        index=0,
        help="Optional override for the required target role field.",
    )

    if st.button("Check API health"):
        try:
            response = requests.get(f"{api_base_url}/health", timeout=10)
            response.raise_for_status()
            st.success(response.json())
        except Exception as exc:
            st.error(f"API unavailable: {exc}")

uploaded_file = st.file_uploader("Upload a resume PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process resume"):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"target_role": target_role} if target_role else {}
            response = requests.post(f"{api_base_url}/process", files=files, data=data, timeout=120)
            response.raise_for_status()
            result = response.json()

            st.session_state["last_result"] = result
            st.success("Resume processed successfully")
        except Exception as exc:
            st.error(f"Processing failed: {exc}")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        st.subheader("Result")
        st.json(result)

        if result.get("validated_profile"):
            st.subheader("Validated profile")
            st.json(result["validated_profile"])
        elif result.get("validation_error"):
            st.warning(result["validation_error"])
