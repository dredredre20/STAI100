import json
import requests
import streamlit as st

st.set_page_config(page_title="STAI100 Career Readiness", page_icon="📄", layout="centered")
st.title("📄 STAI100 Career Readiness Advisor")
st.caption("Upload a resume PDF and I'll pull out your profile details.")

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = "http://127.0.0.1:8000"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_file_bytes" not in st.session_state:
    st.session_state["pending_file_bytes"] = None
if "pending_file_name" not in st.session_state:
    st.session_state["pending_file_name"] = None
if "awaiting_role_retry" not in st.session_state:
    st.session_state["awaiting_role_retry"] = False

with st.sidebar:
    st.header("Settings")
    st.session_state["api_base_url"] = st.text_input(
        "FastAPI base URL", value=st.session_state["api_base_url"]
    )
    if st.button("Check API health"):
        try:
            response = requests.get(f"{st.session_state['api_base_url']}/health", timeout=10)
            response.raise_for_status()
            st.success(response.json())
        except Exception as exc:
            st.error(f"API unavailable: {exc}")

    st.divider()
    target_role = st.selectbox(
        "Target role",
        ["", "data_scientist", "cloud_engineering"],
        index=0,
        help="Optional — set this upfront to skip the follow-up question, "
             "or leave blank and I'll ask you to pick one if needed.",
    )

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state["messages"] = []
        st.session_state["pending_file_bytes"] = None
        st.session_state["pending_file_name"] = None
        st.session_state["awaiting_role_retry"] = False
        st.rerun()


def format_profile_summary(profile: dict) -> str:
    lines = [
        f"**Target role:** {profile.get('target_role')}",
        f"**Current role:** {profile.get('current_role_category') or 'Not specified'}",
        f"**Years of experience:** {profile.get('years_of_experience')}",
        f"**Education:** {profile.get('education_level') or 'Not specified'}",
    ]
    skills = profile.get("skills") or []
    if skills:
        lines.append(f"**Skills found ({len(skills)}):** " + ", ".join(skills))
    certs = profile.get("certifications") or []
    if certs:
        lines.append("**Certifications:** " + ", ".join(certs))
    return "\n\n".join(lines)


def stream_process_resume(file_bytes: bytes, file_name: str, target_role: str | None, status_box, response_placeholder):
    """Consumes the /process/stream SSE endpoint. Same consumption pattern
    as stream_faq_bot in the Oakridge lab: iterate over the response's
    streamed lines, parse each "data: ..." event, and update the UI as
    events arrive — except here most events are STAGE labels (shown via the
    status box) rather than text tokens (shown via the placeholder), since
    the pipeline produces structured data, not prose, until the very end."""
    files = {"file": (file_name, file_bytes, "application/pdf")}
    data = {"target_role": target_role} if target_role else {}

    final_result = None

    with requests.post(
        f"{st.session_state['api_base_url']}/process/stream",
        files=files, data=data, stream=True, timeout=180,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break

            event = json.loads(payload)
            if event["type"] == "stage":
                status_box.update(label=event["label"], state="running")
            elif event["type"] == "result":
                final_result = event["result"]
                status_box.update(label="Done", state="complete")
            elif event["type"] == "error":
                status_box.update(label="Something went wrong", state="error")
                response_placeholder.markdown(f"Error: {event['error']}")

    return final_result


# ── Render chat history ──────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_file = st.file_uploader(
    "Upload a resume PDF", type=["pdf"], label_visibility="collapsed"
)

if uploaded_file is not None and st.session_state["pending_file_bytes"] is None:
    st.session_state["pending_file_bytes"] = uploaded_file.getvalue()
    st.session_state["pending_file_name"] = uploaded_file.name

    st.session_state["messages"].append(
        {"role": "user", "content": f"📎 Uploaded resume: **{uploaded_file.name}**"}
    )
    with st.chat_message("user"):
        st.markdown(f"📎 Uploaded resume: **{uploaded_file.name}**")

    with st.chat_message("assistant"):
        status_box = st.status("Reading resume...", expanded=True)
        response_placeholder = st.empty()

        try:
            result = stream_process_resume(
                st.session_state["pending_file_bytes"],
                st.session_state["pending_file_name"],
                target_role=target_role if target_role else None,
                status_box=status_box,
                response_placeholder=response_placeholder,
            )

            if result and result.get("needs_clarification"):
                answer_text = (
                    "I've read through your resume, but I couldn't tell which role "
                    "you're aiming for. Please select a **Target role** from the "
                    "dropdown in the sidebar, then click the button below."
                )
                response_placeholder.markdown(answer_text)
                st.session_state["messages"].append({"role": "assistant", "content": answer_text})
                # Keep pending_file_bytes/pending_file_name set — the retry
                # button below reuses them, so the user doesn't have to
                # re-select the file (Streamlit's file_uploader won't
                # re-trigger on the same file being picked again anyway).
                st.session_state["awaiting_role_retry"] = True

            elif result and result.get("is_complete"):
                profile = result["validated_profile"]
                answer_text = "Here's what I found in your resume:\n\n" + format_profile_summary(profile)
                response_placeholder.markdown(answer_text)
                st.session_state["messages"].append({"role": "assistant", "content": answer_text})
                st.session_state["pending_file_bytes"] = None
                st.session_state["pending_file_name"] = None
                st.session_state["awaiting_role_retry"] = False

            else:
                error_text = f"Sorry, I couldn't process that resume: {result.get('validation_error') if result else 'Unknown error'}"
                response_placeholder.markdown(error_text)
                st.session_state["messages"].append({"role": "assistant", "content": error_text})
                st.session_state["pending_file_bytes"] = None
                st.session_state["pending_file_name"] = None
                st.session_state["awaiting_role_retry"] = False

        except Exception as exc:
            status_box.update(label="Request failed", state="error")
            error_text = f"Something went wrong talking to the server: {exc}"
            response_placeholder.markdown(error_text)
            st.session_state["messages"].append({"role": "assistant", "content": error_text})
            st.session_state["pending_file_bytes"] = None
            st.session_state["pending_file_name"] = None



# ── Retry button — reuses the already-uploaded file once a target role ──
# has been picked from the sidebar dropdown, so the user isn't forced to
# re-select the same file (Streamlit's file_uploader won't re-fire an
# on-upload action for an unchanged file selection anyway).
if st.session_state["awaiting_role_retry"] and st.session_state["pending_file_bytes"] is not None:
    if not target_role:
        st.info("Select a target role in the sidebar to enable the retry button.")
    else:
        if st.button(f"Retry with target role: {target_role}"):
            with st.chat_message("assistant"):
                status_box = st.status("Thinking...", expanded=True)
                response_placeholder = st.empty()
                try:
                    result = stream_process_resume(
                        st.session_state["pending_file_bytes"],
                        st.session_state["pending_file_name"],
                        target_role=target_role,
                        status_box=status_box,
                        response_placeholder=response_placeholder,
                    )
                    if result and result.get("is_complete"):
                        profile = result["validated_profile"]
                        answer_text = "Got it — here's your full profile:\n\n" + format_profile_summary(profile)
                        response_placeholder.markdown(answer_text)
                        st.session_state["messages"].append({"role": "assistant", "content": answer_text})
                    else:
                        error_text = f"Still couldn't complete processing: {result.get('validation_error') if result else 'Unknown error'}"
                        response_placeholder.markdown(error_text)
                        st.session_state["messages"].append({"role": "assistant", "content": error_text})
                finally:
                    st.session_state["pending_file_bytes"] = None
                    st.session_state["pending_file_name"] = None
                    st.session_state["awaiting_role_retry"] = False