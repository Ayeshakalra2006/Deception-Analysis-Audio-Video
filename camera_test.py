
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Camera and Mic Test")

st.title("Camera and Microphone Test")

st.write("Allow camera and microphone permissions when asked.")

webrtc_streamer(
    key="camera-mic-test",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": True,
        "audio": True
    },
    async_processing=True
)
