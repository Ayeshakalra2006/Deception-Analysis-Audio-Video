
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = PROJECT_DIR / "data" / "processed" / "audio"
METADATA_PATH = PROJECT_DIR / "data" / "metadata.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(METADATA_PATH)

features = []
labels = []
video_names = []

for index, row in df.iterrows():

    video_name = Path(row["video_path"]).stem
    audio_path = AUDIO_DIR / f"{video_name}.wav"

    if not audio_path.exists():
        print("Audio missing:", video_name)
        continue

    try:
        audio, sr = librosa.load(
            str(audio_path),
            sr=16000
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        # Mean and standard deviation of MFCCs
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        combined_features = np.concatenate(
            [mfcc_mean, mfcc_std]
        )

        features.append(combined_features)
        labels.append(row["label_id"])
        video_names.append(video_name)

    except Exception as e:
        print("Error:", video_name, e)

    print(f"Processed: {index + 1}/{len(df)}")

X = np.array(features, dtype=np.float32)
y = np.array(labels, dtype=np.int64)

np.savez_compressed(
    OUTPUT_DIR / "audio_features.npz",
    X=X,
    y=y,
    video_names=np.array(video_names)
)

print("\nAUDIO FEATURES EXTRACTED!")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)
print("Saved in:", OUTPUT_DIR / "audio_features.npz")
