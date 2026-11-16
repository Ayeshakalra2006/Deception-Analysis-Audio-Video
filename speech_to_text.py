
import asyncio
import os
import tempfile

import edge_tts
from faster_whisper import WhisperModel


# Load speech recognition model only once
_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        _whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )

    return _whisper_model


# Convert recorded speech into text
def transcribe_audio(audio_path):

    model = get_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        language="en",
        vad_filter=True
    )

    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return transcript


# Convert AI interview question into spoken audio
async def generate_speech(text, output_path):

    communicator = edge_tts.Communicate(
        text,
        voice="en-US-AriaNeural"
    )

    await communicator.save(output_path)


def speak_question(text):

    with tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ) as temp_file:

        audio_path = temp_file.name

    try:
        asyncio.run(
            generate_speech(text, audio_path)
        )

        with open(audio_path, "rb") as file:
            audio_bytes = file.read()

        return audio_bytes

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
