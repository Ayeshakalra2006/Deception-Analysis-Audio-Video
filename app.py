
import streamlit as st
import tempfile
import os

from src.predict import predict_video
from interview import run_interview

st.set_page_config(
    page_title="Deception Analysis Using AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE ----------------

PAGES = [
    "🏠 Dashboard",
    "📁 Upload Analysis",
    "🎤 AI Interview"
]

if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

if "navigation" not in st.session_state:
    st.session_state.navigation = st.session_state.current_page


def open_upload():
    st.session_state.current_page = "📁 Upload Analysis"
    st.session_state.navigation = "📁 Upload Analysis"


def open_interview():
    st.session_state.current_page = "🎤 AI Interview"
    st.session_state.navigation = "🎤 AI Interview"


# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(37,99,235,0.13), transparent 38%),
        radial-gradient(ellipse at 90% 80%, rgba(109,40,217,0.10), transparent 35%),
        #080f1d;
    color: #e2e8f0;
}

.block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: rgba(8,15,29,0.95);
    border-bottom: 1px solid #24334d;
}

[data-testid="stToolbar"] {
    background: transparent;
}

[data-testid="stDeployButton"] button {
    background: linear-gradient(135deg, #2563eb, #7045d6) !important;
    color: white !important;
    border: 1px solid #6476ef !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

[data-testid="stDeployButton"] button:hover {
    background: linear-gradient(135deg, #3478f6, #8055e8) !important;
    border-color: #9aa9ff !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111d32 0%, #0c1628 100%);
    border-right: 1px solid #293b59;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

[data-testid="stSidebar"] h1 {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px;
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #91a4c4 !important;
    letter-spacing: 1.2px;
    font-size: 0.68rem;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 9px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.035);
    border: 1px solid #263852;
    border-radius: 11px;
    padding: 12px 13px;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(59,130,246,0.13);
    border-color: #5278bd;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(100deg, rgba(37,99,235,0.30), rgba(109,40,217,0.23));
    border: 1px solid #557bea;
}

[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #e2e8f0 !important;
    font-weight: 600;
}

h1, h2, h3 {
    color: #f8fafc !important;
    letter-spacing: -0.4px;
}

h1 {
    font-weight: 800 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

p, label {
    color: #bdcbe0;
}

hr {
    border-color: #263852 !important;
    opacity: 0.8;
}

div.stButton > button {
    min-height: 47px;
    border-radius: 10px;
    border: 1px solid #344968;
    background: #17243a;
    color: #eaf1ff;
    font-weight: 650;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    border-color: #6488e9;
    color: white;
    background: #203454;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(110deg, #2563eb, #7045d6);
    color: white;
    border: none;
    box-shadow: 0 5px 18px rgba(37,99,235,0.18);
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(110deg, #3478f6, #8055e8);
    box-shadow: 0 7px 22px rgba(37,99,235,0.30);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(20,32,53,0.75);
    border: 1px solid #2a3d5c;
    border-radius: 15px;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17263e, #111d31);
    border: 1px solid #304565;
    border-radius: 13px;
    padding: 17px;
}

[data-testid="stMetricLabel"] {
    color: #9db2d3 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 750;
}

[data-testid="stFileUploader"] {
    background: rgba(22,36,59,0.55);
    border: 1px dashed #526d98;
    border-radius: 13px;
    padding: 12px;
}

[data-testid="stVideo"] {
    border: 1px solid #304565;
    border-radius: 13px;
    overflow: hidden;
}

[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #2563eb, #8b5cf6);
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR NAVIGATION ----------------

with st.sidebar:
    st.title("🎭 DECEPTION AI")
    st.caption("MULTIMODAL ANALYSIS PLATFORM")

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        PAGES,
        key="navigation"
    )

    st.session_state.current_page = selected_page

    st.divider()
    st.caption("AUDIO  •  VIDEO  •  AI")


# ---------------- DASHBOARD ----------------

if st.session_state.current_page == "🏠 Dashboard":

    st.title("Deception Analysis Using AI")

    st.write(
        "An intelligent multimodal platform for analysing "
        "audio and video using custom-trained AI models."
    )

    st.divider()

    st.subheader("Platform Overview")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("📁 Upload Analysis")

            st.write(
                "Analyse an existing video using audio and "
                "visual features extracted by our trained model."
            )

            st.button(
                "Open Upload Analysis →",
                type="primary",
                use_container_width=True,
                on_click=open_upload
            )

    with col2:
        with st.container(border=True):
            st.subheader("🎤 AI Interview")

            st.write(
                "Participate in an AI-powered interview using "
                "your camera and microphone."
            )

            st.button(
                "Open AI Interview →",
                type="primary",
                use_container_width=True,
                on_click=open_interview
            )

    st.divider()

    st.subheader("Technology Stack")

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        with st.container(border=True):
            st.subheader("🔊 Audio Analysis")
            st.write("MFCC feature extraction and ANN-based analysis.")

    with c2:
        with st.container(border=True):
            st.subheader("🎥 Video Analysis")
            st.write("Video frame processing and custom CNN.")

    with c3:
        with st.container(border=True):
            st.subheader("🧠 Multimodal AI")
            st.write("Combined audio-video feature analysis.")


# ---------------- UPLOAD ANALYSIS ----------------

elif st.session_state.current_page == "📁 Upload Analysis":

    st.title("📁 Video Analysis")

    st.write(
        "Upload a video to analyse it using "
        "the trained multimodal model."
    )

    st.divider()

    with st.container(border=True):
        st.subheader("Upload Your Video")

        st.caption("Supported formats: MP4, AVI, MOV, MKV")

        uploaded_file = st.file_uploader(
            "Choose video",
            type=["mp4", "avi", "mov", "mkv"]
        )

    if uploaded_file is not None:

        st.success("Video uploaded successfully!")

        st.video(uploaded_file)

        if st.button(
            "🔍 Analyse Video",
            type="primary",
            use_container_width=True
        ):

            temp_path = None

            try:
                extension = os.path.splitext(uploaded_file.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                with st.spinner("Analysing audio and video..."):

                    prediction, confidence = predict_video(temp_path)

                st.divider()

                st.subheader("Analysis Result")

                if prediction == "Deceptive":
                    st.error("⚠️ DECEPTIVE")
                else:
                    st.success("✅ TRUTHFUL")

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(
                    max(0.0, min(float(confidence) / 100, 1.0))
                )

            except Exception as e:
                st.error(f"Analysis failed: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    else:
        st.info("Upload a video to begin analysis.")


# ---------------- AI INTERVIEW ----------------

elif st.session_state.current_page == "🎤 AI Interview":

    run_interview()


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "DECEPTION AI  |  MULTIMODAL AUDIO & VIDEO ANALYSIS"
)
