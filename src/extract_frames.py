
import cv2
import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_DIR / "data" / "metadata.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(METADATA_PATH)

FRAMES_PER_VIDEO = 10
IMAGE_SIZE = (224, 224)

success = 0
failed = 0

for index, row in df.iterrows():

    video_path = Path(row["video_path"])
    label = row["label"]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Could not open:", video_path.name)
        failed += 1
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        failed += 1
        cap.release()
        continue

    frame_indices = np.linspace(
        0, total_frames - 1,
        FRAMES_PER_VIDEO,
        dtype=int
    )

    video_output = OUTPUT_DIR / label / video_path.stem
    video_output.mkdir(parents=True, exist_ok=True)

    saved = 0

    for i, frame_index in enumerate(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, IMAGE_SIZE)

            output_path = video_output / f"frame_{i+1}.jpg"

            cv2.imwrite(str(output_path), frame)
            saved += 1

    cap.release()

    if saved > 0:
        success += 1
    else:
        failed += 1

    print(f"Processed video: {index + 1}/{len(df)}")

print("\nFRAME EXTRACTION COMPLETE!")
print("Successful videos:", success)
print("Failed videos:", failed)
print("Frames per video:", FRAMES_PER_VIDEO)
print("Saved in:", OUTPUT_DIR)
