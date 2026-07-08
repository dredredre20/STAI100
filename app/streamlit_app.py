import json
import requests
import streamlit as st

st.set_page_config(page_title="STAI100 Career Readiness", page_icon="📄", layout="centered")
st.title("📄 STAI100 Career Readiness Advisor")
st.caption("Upload a resume PDF and I'll pull out your profile details.")

# Initialize session state variables if they don't exist yet
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
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "processed_file_id" not in st.session_state:
    st.session_state["processed_file_id"] = None


# UI Sidebar for settings and target role
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

# function to format the profile summary for display (confirmation of pdf processing)
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


# function for streaming the resume processing stages from the backend API
def stream_process_resume(file_bytes: bytes, file_name: str, target_role: str | None, status_box, response_placeholder):
    """Consumes the /process/stream SSE endpoint."""
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


# Render chat history 
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_file = st.file_uploader(
    "Upload a resume PDF", type=["pdf"], label_visibility="collapsed",
    key=f"uploader_{st.session_state['uploader_key']}",
)

# if pdf file is uploaded, process it and display results
if uploaded_file is not None and uploaded_file.file_id != st.session_state["processed_file_id"]:
    st.session_state["pending_file_bytes"] = uploaded_file.getvalue()
    st.session_state["pending_file_name"] = uploaded_file.name
    st.session_state["processed_file_id"] = uploaded_file.file_id

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

            # confirmation on the need for target role
            if result and result.get("needs_clarification"):
                answer_text = (
                    "I've read through your resume, but I couldn't tell which role "
                    "you're aiming for. Please select a **Target role** from the "
                    "dropdown in the sidebar, then click the button below."
                )
                response_placeholder.markdown(answer_text)
                st.session_state["messages"].append({"role": "assistant", "content": answer_text})
                st.session_state["awaiting_role_retry"] = True

            # if the resume was processed successfully, display the profile summary
            elif result and result.get("is_complete"):
                profile = result["validated_profile"]
                answer_text = "Here's what I found in your resume:\n\n" + format_profile_summary(profile)
                response_placeholder.markdown(answer_text)
                st.session_state["messages"].append({"role": "assistant", "content": answer_text})
                st.session_state["pending_file_bytes"] = None
                st.session_state["pending_file_name"] = None
                st.session_state["awaiting_role_retry"] = False
                st.session_state["profile_context"] = {
                    "session_id": result.get("session_id", "unset-session-id"),
                    "resume_skills": profile.get("skills", []),
                    "target_role": profile.get("target_role"),
                }

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
                        st.session_state["profile_context"] = {
                            "session_id": result.get("session_id", "unset-session-id"),
                            "resume_skills": profile.get("skills", []),
                            "target_role": profile.get("target_role"),
                        }
                    else:
                        error_text = f"Still couldn't complete processing: {result.get('validation_error') if result else 'Unknown error'}"
                        response_placeholder.markdown(error_text)
                        st.session_state["messages"].append({"role": "assistant", "content": error_text})
                finally:
                    st.session_state["pending_file_bytes"] = None
                    st.session_state["pending_file_name"] = None
                    st.session_state["awaiting_role_retry"] = False

# ── Advisor chat — appears once a profile has been successfully processed ──
if "profile_context" not in st.session_state:
    st.session_state["profile_context"] = None  # {session_id, resume_skills, target_role}
if "advisor_messages" not in st.session_state:
    st.session_state["advisor_messages"] = []

# render the chat adviser interface if profile is available
if st.session_state["profile_context"] is not None:
    st.divider()
    st.subheader("💬 Ask your advisor")
    st.caption("Ask about your skill gaps, progress, or readiness for your target role.")

    for msg in st.session_state["advisor_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    advisor_input = st.chat_input("e.g. 'What skills am I missing?' or 'How is my readiness score?'")
    if advisor_input:
        st.session_state["advisor_messages"].append({"role": "user", "content": advisor_input})
        with st.chat_message("user"):
            st.markdown(advisor_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    ctx = st.session_state["profile_context"]
                    response = requests.post(
                        f"{st.session_state['api_base_url']}/chat",
                        json={
                            "message": advisor_input,
                            "session_id": ctx["session_id"],
                            "resume_skills": ctx["resume_skills"],
                            "target_role": ctx["target_role"],
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    answer = response.json()["answer"]
                except Exception as exc:
                    answer = f"Something went wrong talking to the advisor: {exc}"
                st.markdown(answer)
                st.session_state["advisor_messages"].append({"role": "assistant", "content": answer})
else:
    st.divider()
    st.info("Please upload and extract your resume details to unlock interactive chat assistance.")