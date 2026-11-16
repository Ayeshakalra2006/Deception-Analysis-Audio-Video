
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
FRAME_DIR = BASE_DIR / "data" / "processed" / "frames"
METADATA_PATH = BASE_DIR / "data" / "metadata.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "video_features.npz"

metadata = pd.read_csv(METADATA_PATH)

all_features = []
all_labels = []

for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    video_name = Path(row["video_path"]).stem
    label = row["label"]

    folder = FRAME_DIR / label / video_name
    frame_paths = sorted(folder.glob("*.jpg"))

    frame_features = []

    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path))

        if image is None:
            continue

        image = cv2.resize(image, (224, 224))

        # RGB colour statistics
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mean_rgb = np.mean(image, axis=(0, 1))
        std_rgb = np.std(image, axis=(0, 1))

        # Grayscale statistics
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_gray = np.mean(gray)
        std_gray = np.std(gray)

        # Edge information
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        features = np.concatenate([
            mean_rgb,
            std_rgb,
            [mean_gray, std_gray, edge_density]
        ])

        frame_features.append(features)

    if len(frame_features) == 0:
        print("No frames found:", video_name)
        continue

    # Average and standard deviation across frames
    frame_features = np.array(frame_features)

    video_feature = np.concatenate([
        np.mean(frame_features, axis=0),
        np.std(frame_features, axis=0)
    ])

    all_features.append(video_feature)
    all_labels.append(row["label_id"])

np.savez(
    OUTPUT_PATH,
    X=np.array(all_features, dtype=np.float32),
    y=np.array(all_labels, dtype=np.int64)
)

print("\nVideo feature extraction complete!")
print("Videos processed:", len(all_features))
print("Feature shape:", np.array(all_features).shape)
print("Saved at:", OUTPUT_PATH)
