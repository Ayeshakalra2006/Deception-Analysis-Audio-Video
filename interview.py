
import streamlit as st
import tempfile
import os
import time
import subprocess

from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from interview_config import QUESTIONS
from speech_to_text import speak_question, transcribe_audio
from facts import extract_facts, detect_contradictions
from scoring import calculate_session_score
from report import display_report
from src.predict import predict_video


# --------------------------------------------------
# PAGE DESIGN
# --------------------------------------------------

def apply_design():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #101827 0%, #17243a 55%, #101827 100%);
        color: #f1f5f9;
    }
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.5px;
    }
    p, label, span {
        color: #dbe4f0;
    }
    .hero-card {
        background: linear-gradient(120deg, #243b55, #141e30);
        border: 1px solid #354a65;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.2);
    }
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 15px;
        color: #bdcce0;
    }
    .question-card {
        background: #1d2b40;
        border: 1px solid #354b66;
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0 22px 0;
    }
    .question-label {
        color: #60a5fa;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .question-text {
        color: #ffffff;
        font-size: 23px;
        font-weight: 650;
        margin-top: 12px;
        line-height: 1.5;
    }
    .section-card {
        background: #1d2b40;
        border: 1px solid #354b66;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
    }
    .result-truthful {
        background: rgba(22, 163, 74, 0.13);
        border: 1px solid #16a34a;
        border-radius: 15px;
        padding: 22px;
        margin: 15px 0;
    }
    .result-deceptive {
        background: rgba(220, 38, 38, 0.13);
        border: 1px solid #dc2626;
        border-radius: 15px;
        padding: 22px;
        margin: 15px 0;
    }
    .result-title {
        font-size: 25px;
        font-weight: 800;
        color: white;
    }
    .result-subtitle {
        color: #cbd5e1;
        font-size: 14px;
        margin-top: 6px;
    }
    div.stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 700;
        border: 1px solid #526780;
        transition: 0.2s ease;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        border-color: #60a5fa;
        transform: translateY(-1px);
    }
    [data-testid="stMetric"] {
        background: #1d2b40;
        border: 1px solid #354b66;
        border-radius: 12px;
        padding: 15px;
    }
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    hr {
        border-color: #354b66;
    }
    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# RECORDING FILE
# --------------------------------------------------

def create_recording_path():
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    os.remove(path)
    return path


def video_has_readable_frames(video_path):
    try:
        import cv2

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            capture.release()
            return False

        success, frame = capture.read()
        capture.release()

        return bool(
            success and
            frame is not None and
            frame.size > 0
        )

    except Exception:
        return False


def prepare_video_for_prediction(video_path):
    """
    Checks whether the recorded video has readable frames.
    If not, attempts to convert it into a standard MP4.
    """

    if video_has_readable_frames(video_path):
        return video_path

    converted_path = None

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        fd, converted_path = tempfile.mkstemp(
            suffix="_compatible.mp4"
        )
        os.close(fd)
        os.remove(converted_path)

        command = [
            ffmpeg_exe,
            "-y",
            "-i", video_path,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            converted_path
        ]

        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )

        if (
            completed.returncode == 0
            and os.path.exists(converted_path)
            and os.path.getsize(converted_path) > 1000
            and video_has_readable_frames(converted_path)
        ):
            return converted_path

        error_text = (completed.stderr or "")[-1200:]

        raise RuntimeError(
            "Recorded video could not be decoded or converted. "
            "Please record again.\n" + error_text
        )

    except Exception:
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except OSError:
                pass
        raise


# --------------------------------------------------
# SESSION INITIALIZATION
# --------------------------------------------------

def initialize_state():

    defaults = {
        "interview_active": False,
        "interview_completed": False,
        "question_index": 0,
        "answers": {},
        "model_results": [],
        "current_contradictions": [],
        "question_audio_cache": {},
        "recording_path": None,
        "recording_question_index": None,
        "recording_started": False,
        "analysis_done": False,
        "current_result": None,
        "recording_ready": False,
        "recording_stopped_at": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------
# LIVE CONTRADICTION ANALYSIS
# --------------------------------------------------

def update_contradictions():

    answers = st.session_state.answers

    facts = extract_facts(answers)

    contradictions = detect_contradictions(facts)

    st.session_state.current_contradictions = contradictions

    return contradictions


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

def cleanup_recording():

    path = st.session_state.get("recording_path")

    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------------------------
# START NEW INTERVIEW
# --------------------------------------------------

def start_new_session():

    cleanup_recording()

    st.session_state.interview_active = True
    st.session_state.interview_completed = False
    st.session_state.question_index = 0

    st.session_state.answers = {}
    st.session_state.model_results = []
    st.session_state.current_contradictions = []
    st.session_state.question_audio_cache = {}

    st.session_state.recording_path = None
    st.session_state.recording_question_index = None

    st.session_state.recording_started = False
    st.session_state.recording_ready = False
    st.session_state.recording_stopped_at = None

    st.session_state.analysis_done = False
    st.session_state.current_result = None


# --------------------------------------------------
# NEXT QUESTION
# --------------------------------------------------

def move_to_next_question():

    cleanup_recording()

    st.session_state.question_index += 1

    st.session_state.recording_path = None
    st.session_state.recording_question_index = None

    st.session_state.recording_started = False
    st.session_state.recording_ready = False
    st.session_state.recording_stopped_at = None

    st.session_state.analysis_done = False
    st.session_state.current_result = None

    if st.session_state.question_index >= len(QUESTIONS):
        st.session_state.interview_active = False
        st.session_state.interview_completed = True

    st.rerun()


# --------------------------------------------------
# RETRY ANSWER
# --------------------------------------------------

def retry_current_answer():

    question_id = QUESTIONS[
        st.session_state.question_index
    ]["id"]

    st.session_state.answers.pop(question_id, None)

    st.session_state.model_results = [
        result
        for result in st.session_state.model_results
        if result.get("question_id") != question_id
    ]

    update_contradictions()

    cleanup_recording()

    st.session_state.recording_path = None
    st.session_state.recording_question_index = None

    st.session_state.recording_started = False
    st.session_state.recording_ready = False
    st.session_state.recording_stopped_at = None

    st.session_state.analysis_done = False
    st.session_state.current_result = None

    st.rerun()


# --------------------------------------------------
# ANALYZE ANSWER
# --------------------------------------------------

def analyze_current_answer(
    recording_path,
    question_text,
    question_id
):

    if not os.path.exists(recording_path):
        raise ValueError(
            "Recording file was not created."
        )

    if os.path.getsize(recording_path) < 1000:
        raise ValueError(
            "Recording file is empty or too small. Please record again."
        )

    # Transcribe the original recording.
    transcript = transcribe_audio(recording_path)

    if not transcript or not transcript.strip():
        raise ValueError(
            "No speech detected. Please retry."
        )

    # Prepare a readable video for the existing model.
    prediction_path = prepare_video_for_prediction(
        recording_path
    )

    try:
        prediction, confidence = predict_video(
            prediction_path
        )

    finally:
        if (
            prediction_path != recording_path
            and os.path.exists(prediction_path)
        ):
            try:
                os.remove(prediction_path)
            except OSError:
                pass

    if prediction == "Deceptive":
        risk = confidence / 100
    else:
        risk = (100 - confidence) / 100

    return {
        "question": question_text,
        "question_id": question_id,
        "transcript": transcript,
        "prediction": prediction,
        "confidence": confidence,
        "risk": risk
    }


# --------------------------------------------------
# INTERVIEW PAGE
# --------------------------------------------------

def run_interview():

    initialize_state()
    apply_design()

    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">🎭 AI Deception Analysis</div>
        <div class="hero-subtitle">
            Multimodal Interview System · Audio & Video Analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # START SCREEN

    if (
        not st.session_state.interview_active
        and not st.session_state.interview_completed
    ):

        st.markdown("""
        <div class="section-card">
            <h3>Welcome to the AI Interview</h3>
            <p>
                Answer a series of questions using your camera
                and microphone. Each response will be analysed
                before you proceed to the next question.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "🚀 Start Interview",
            type="primary",
            use_container_width=True
        ):
            start_new_session()
            st.rerun()

        return

    # ACTIVE INTERVIEW

    if st.session_state.interview_active:

        index = st.session_state.question_index

        if index < len(QUESTIONS):

            question_data = QUESTIONS[index]

            question_id = question_data["id"]
            question_text = question_data["question"]

            total = len(QUESTIONS)

            st.progress((index + 1) / total)

            st.caption(
                f"QUESTION {index + 1} OF {total}"
            )

            st.markdown(f"""
            <div class="question-card">
                <div class="question-label">Interview Question</div>
                <div class="question-text">{question_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # INTERVIEWER VOICE

            if question_id not in st.session_state.question_audio_cache:

                with st.spinner("Preparing interviewer voice..."):

                    try:
                        audio_bytes = speak_question(
                            question_text
                        )

                        st.session_state.question_audio_cache[
                            question_id
                        ] = audio_bytes

                    except Exception as e:
                        st.error(
                            f"Voice generation failed: {e}"
                        )
                        return

            st.audio(
                st.session_state.question_audio_cache[
                    question_id
                ],
                format="audio/mp3"
            )

            # CREATE RECORDING

            if (
                st.session_state.recording_path is None
                or st.session_state.recording_question_index != index
            ):

                st.session_state.recording_path = (
                    create_recording_path()
                )

                st.session_state.recording_question_index = index

                st.session_state.recording_started = False
                st.session_state.recording_ready = False
                st.session_state.recording_stopped_at = None

            recording_path = st.session_state.recording_path

            # CAMERA AND MICROPHONE

            if not st.session_state.analysis_done:

                def recorder_factory():

                    return MediaRecorder(
                        recording_path,
                        format="mp4"
                    )

                st.markdown("""
                <div class="section-card">
                    <h3>📹 Your Recording</h3>
                    <p>
                        Enable your camera and microphone
                        to record your answer.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                ctx = webrtc_streamer(
                    key=f"interview-question-{index}",

                    mode=WebRtcMode.SENDRECV,

                    media_stream_constraints={
                        "video": True,
                        "audio": True
                    },

                    in_recorder_factory=recorder_factory,

                    rtc_configuration={
                        "iceServers": [
                            {
                                "urls": [
                                    "stun:stun.l.google.com:19302",
                                    "stun:stun1.l.google.com:19302"
                                ]
                            }
                        ]
                    },

                    async_processing=True
                )

                if ctx.state.playing:

                    st.session_state.recording_started = True
                    st.session_state.recording_ready = False

                    st.info(
                        "🔴 Recording in progress. Speak your answer."
                    )

                elif st.session_state.recording_started:

                    if st.session_state.recording_stopped_at is None:
                        st.session_state.recording_stopped_at = time.time()

                    elapsed = (
                        time.time()
                        - st.session_state.recording_stopped_at
                    )

                    if elapsed < 3:

                        st.info(
                            "Finalizing your video recording..."
                        )

                        time.sleep(0.5)
                        st.rerun()

                    if os.path.exists(recording_path):

                        file_size = os.path.getsize(
                            recording_path
                        )

                        if file_size > 1000:

                            st.session_state.recording_ready = True

                            st.success(
                                "✅ Recording saved successfully."
                            )

                        elif elapsed < 8:

                            st.info(
                                "Saving your recording. Please wait..."
                            )

                            time.sleep(0.5)
                            st.rerun()

                        else:

                            st.error(
                                "Recording is empty. Please record again."
                            )

                    elif elapsed < 8:

                        st.info(
                            "Waiting for the recording file to finish saving..."
                        )

                        time.sleep(0.5)
                        st.rerun()

                    else:

                        st.error(
                            "Recording file was not created. Please record again."
                        )

                else:

                    st.info(
                        "Waiting for camera and microphone..."
                    )

            # RECORDING PREVIEW AND ANALYSIS

            if (
                st.session_state.recording_ready
                and not st.session_state.analysis_done
            ):

                st.subheader("Recorded Answer Preview")

                st.video(recording_path)

                if st.button(
                    "🔍 Analyze My Answer",
                    type="primary",
                    use_container_width=True,
                    key=f"analyze_{index}"
                ):

                    try:

                        with st.spinner(
                            "Analysing your audio and video..."
                        ):

                            result = analyze_current_answer(
                                recording_path,
                                question_text,
                                question_id
                            )

                        st.session_state.current_result = result

                        st.session_state.answers[
                            question_id
                        ] = result["transcript"]

                        st.session_state.model_results = [
                            item for item in st.session_state.model_results
                            if item.get("question_id") != question_id
                        ]

                        st.session_state.model_results.append(result)

                        update_contradictions()

                        st.session_state.analysis_done = True

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Analysis failed: {e}"
                        )

                if st.button(
                    "🔄 Record Again",
                    use_container_width=True,
                    key=f"retry_recording_{index}"
                ):

                    retry_current_answer()

            # ANALYSIS RESULT

            if st.session_state.analysis_done:

                result = st.session_state.current_result

                st.divider()

                st.header("📊 Analysis Result")

                st.markdown("**Your Answer:**")

                st.markdown(
                    f"""
                    <div class="section-card">
                        {result["transcript"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if result["prediction"] == "Deceptive":

                    st.markdown("""
                    <div class="result-deceptive">
                        <div class="result-title">⚠️ DECEPTIVE</div>
                        <div class="result-subtitle">
                            The model classified this response as deceptive.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="result-truthful">
                        <div class="result-title">✅ TRUTHFUL</div>
                        <div class="result-subtitle">
                            The model classified this response as truthful.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.metric(
                    "Model Confidence",
                    f"{result['confidence']:.2f}%"
                )

                # CROSS-QUESTION CONSISTENCY CHECK

                st.divider()

                st.subheader(
                    "🔎 Cross-Question Consistency Check"
                )

                current_contradictions = (
                    st.session_state.current_contradictions
                )

                if current_contradictions:

                    st.warning(
                        "Potential inconsistencies were found "
                        "between your submitted answers:"
                    )

                    for item in current_contradictions:
                        st.warning(item)

                else:

                    st.success(
                        "No contradictions detected among the answers "
                        "submitted so far."
                    )

                st.caption(
                    "This check compares your answers across questions. "
                    "An inconsistency is not proof of deception."
                )

                st.divider()

                st.info(
                    "Review your result before moving to the next question."
                )

                col1, col2 = st.columns(2)

                with col1:

                    if st.button(
                        "🔄 Retry Answer",
                        use_container_width=True,
                        key=f"retry_result_{index}"
                    ):
                        retry_current_answer()

                with col2:

                    if st.button(
                        "Next Question ➜",
                        type="primary",
                        use_container_width=True,
                        key=f"next_{index}"
                    ):
                        move_to_next_question()

    # FINAL REPORT

    if st.session_state.interview_completed:

        st.markdown("""
        <div class="hero-card">
            <div class="hero-title">🎉 Interview Completed</div>
            <div class="hero-subtitle">
                Your interview responses have been processed.
            </div>
        </div>
        """, unsafe_allow_html=True)

        answers = st.session_state.answers

        contradictions = update_contradictions()

        results = st.session_state.model_results

        if results:

            average_risk = sum(
                item["risk"] for item in results
            ) / len(results)

        else:

            average_risk = 0.50

        score_result = calculate_session_score(
            contradictions=contradictions,
            voice_deviation=0.0,
            model_score=average_risk
        )

        display_report(
            answers=answers,
            contradictions=contradictions,
            score_result=score_result
        )

        st.header("Individual Answer Analysis")

        for i, result in enumerate(results, start=1):

            with st.expander(f"Question {i}"):

                st.write(result["question"])

                st.write(
                    "**Transcript:**",
                    result["transcript"]
                )

                st.write(
                    "**Prediction:**",
                    result["prediction"]
                )

                st.write(
                    f"**Confidence:** {result['confidence']:.2f}%"
                )

        if st.button(
            "🔄 Start New Interview",
            type="primary",
            use_container_width=True
        ):

            start_new_session()
            st.rerun()
