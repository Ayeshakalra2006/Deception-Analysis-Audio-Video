
import streamlit as st
import tempfile
import os

from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from src.predict import predict_video


st.set_page_config(page_title="Automatic Deception Analysis")

st.title("AI-Based Automatic Deception Analysis")


# Initialize session state
if "recording_path" not in st.session_state:

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    os.remove(path)

    st.session_state["recording_path"] = path

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False


# Get path BEFORE creating the background callback
recording_path = st.session_state["recording_path"]


# Do not access Streamlit session_state inside this callback
def recorder_factory():

    return MediaRecorder(
        recording_path,
        format="mp4"
    )


st.write("Start your camera and microphone. Speak for 10-15 seconds.")

ctx = webrtc_streamer(
    key="automatic-analysis",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": True,
        "audio": True
    },
    in_recorder_factory=recorder_factory,
    async_processing=True
)


if ctx.state.playing:

    st.info("Recording your answer...")


elif os.path.exists(recording_path):

    file_size = os.path.getsize(recording_path)

    if file_size > 0:

        if not st.session_state["analysis_done"]:

            with st.spinner("Analysing audio and video..."):

                try:

                    prediction, confidence = predict_video(
                        recording_path
                    )

                    st.session_state["prediction"] = prediction
                    st.session_state["confidence"] = confidence
                    st.session_state["analysis_done"] = True

                except Exception as e:

                    st.error(f"Analysis failed: {e}")

        if st.session_state["analysis_done"]:

            st.header("Analysis Result")

            if st.session_state["prediction"] == "Deceptive":
                st.error("Prediction: DECEPTIVE")
            else:
                st.success("Prediction: TRUTHFUL")

            st.metric(
                "Model Confidence",
                f"{st.session_state['confidence']:.2f}%"
            )

            st.warning(
                "Experimental model prediction only. "
                "This does not prove whether someone is lying."
            )

    else:
        st.info("Recording file is empty or still being written.")

else:

    st.info("Start the camera to begin recording.")
