import requests
import streamlit as st

st.set_page_config(page_title="STAI100 Resume Intake", page_icon="📄", layout="wide")
st.title("STAI100 Resume Intake")
st.caption("Upload a resume PDF and inspect the parsed profile from the local AI pipeline.")

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = "http://127.0.0.1:8000"

with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input("FastAPI base URL", value=st.session_state["api_base_url"])
    st.session_state["api_base_url"] = api_base_url  # persist any change back to session_state

    target_role = st.selectbox(
        "Target role",
        ["", "data_scientist", "cloud_engineering"],
        index=0,
        help="Optional override for the required target role field. Leave blank on your "
             "first attempt — you'll only need to set this if the resume doesn't state "
             "a clear target role and the system asks you to clarify.",
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

            # Only declare success if the pipeline actually completed.
            # needs_clarification=True means the backend deliberately did NOT
            # run the interactive clarification loop (it can't — there's no
            # terminal attached to a web request) and is instead handing the
            # decision back to this frontend.
            if result.get("needs_clarification"):
                st.warning(
                    "This resume doesn't clearly state a target role, so I can't "
                    "finish processing it yet."
                )
            elif result.get("is_complete"):
                st.success("Resume processed successfully")
            else:
                st.error(
                    f"Processing did not complete: "
                    f"{result.get('validation_error') or 'Unknown error'}"
                )
        except Exception as exc:
            st.error(f"Processing failed: {exc}")

if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    # Guide the user to the fix directly, rather than just showing raw JSON
    # and leaving them to figure out what "needs_clarification" means.
    if result.get("needs_clarification"):
        st.subheader("One more thing needed")
        st.write(result.get("clarification_question", "Please specify your target role."))
        st.info(
            "Select a **Target role** in the sidebar, then click "
            "**Process resume** again to resubmit the same file with that role set."
        )

    st.subheader("Result")
    st.json(result)

    if result.get("validated_profile"):
        st.subheader("Validated profile")
        st.json(result["validated_profile"])
    elif result.get("validation_error"):
        st.warning(result["validation_error"])