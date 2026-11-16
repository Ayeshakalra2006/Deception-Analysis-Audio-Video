
import cv2
import numpy as np
import librosa
import torch
import torch.nn as nn
import joblib
import subprocess
import tempfile
import imageio_ffmpeg

from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "multimodal_model.pth"
SCALER_PATH = BASE_DIR / "models" / "multimodal_scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Same architecture used during training
class MultimodalModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)


# Extract audio features
def extract_audio_features(video_path):

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        audio_path = Path(temp.name)

    try:
        subprocess.run(
            [
                ffmpeg, "-y",
                "-i", str(video_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                str(audio_path)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        y, sr = librosa.load(str(audio_path), sr=16000)

        if len(y) == 0:
            raise ValueError("No audio found in uploaded video.")

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        )

        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1)
        ])

        return features

    finally:
        if audio_path.exists():
            audio_path.unlink()


# Extract video features
def extract_video_features(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames.")

    positions = np.linspace(
        0, total_frames - 1, 10, dtype=int
    )

    frame_features = []

    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        success, frame = cap.read()

        if not success:
            continue

        image = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mean_rgb = np.mean(image, axis=(0, 1))
        std_rgb = np.std(image, axis=(0, 1))

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        mean_gray = np.mean(gray)
        std_gray = np.std(gray)

        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        features = np.concatenate([
            mean_rgb,
            std_rgb,
            [mean_gray, std_gray, edge_density]
        ])

        frame_features.append(features)

    cap.release()

    if len(frame_features) == 0:
        raise ValueError("Could not extract video frames.")

    frame_features = np.array(frame_features)

    video_features = np.concatenate([
        np.mean(frame_features, axis=0),
        np.std(frame_features, axis=0)
    ])

    return video_features


# Load model and predict
def predict_video(video_path):

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Trained multimodal model or scaler not found.")

    audio_features = extract_audio_features(video_path)
    video_features = extract_video_features(video_path)

    combined_features = np.concatenate([
        audio_features,
        video_features
    ]).reshape(1, -1)

    scaler = joblib.load(SCALER_PATH)
    combined_features = scaler.transform(combined_features)

    model = MultimodalModel(combined_features.shape[1]).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    tensor = torch.tensor(
        combined_features,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_class = torch.argmax(probabilities).item()

    label = "Deceptive" if predicted_class == 1 else "Truthful"
    confidence = probabilities[predicted_class].item() * 100

    return label, confidence
