
import numpy as np
import librosa


def extract_voice_features(audio_path):
    """
    Extracts basic voice features from an audio recording:
    pitch, loudness, duration and pause ratio.
    """

    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio recording is empty.")

    # Duration of the recording
    duration = len(audio) / sample_rate

    # Estimate pitch
    pitch = librosa.yin(
        audio,
        fmin=65,
        fmax=400,
        sr=sample_rate
    )

    average_pitch = float(np.mean(pitch))

    # Estimate loudness
    rms = librosa.feature.rms(y=audio)[0]
    average_loudness = float(np.mean(rms))

    # Detect non-silent sections
    intervals = librosa.effects.split(
        audio,
        top_db=30
    )

    speech_duration = sum(
        (end - start) / sample_rate
        for start, end in intervals
    )

    pause_ratio = max(
        0.0,
        min(1.0, 1 - (speech_duration / duration))
    )

    return {
        "average_pitch": average_pitch,
        "average_loudness": average_loudness,
        "pause_ratio": pause_ratio,
        "duration": duration
    }


def compare_with_baseline(current, baseline):
    """
    Compares current voice features with the person's
    own baseline features.

    Returns a simple normalized deviation indicator.
    """

    feature_names = [
        "average_pitch",
        "average_loudness",
        "pause_ratio"
    ]

    deviations = []

    for name in feature_names:
        base_value = baseline.get(name, 0.0)
        current_value = current.get(name, 0.0)

        if abs(base_value) < 1e-6:
            deviation = abs(current_value - base_value)
        else:
            deviation = abs(
                current_value - base_value
            ) / abs(base_value)

        deviations.append(min(deviation, 1.0))

    return float(np.mean(deviations))
