
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "Real-life_Deception_Detection_2016"
)

records = []

for label in ["Deceptive", "Truthful"]:

    folder = DATASET_DIR / "Clips" / label

    for video in folder.glob("*.mp4"):

        records.append({
            "video_path": str(video.resolve()),
            "label": label,
            "label_id": 1 if label == "Deceptive" else 0
        })

df = pd.DataFrame(records)

output_path = PROJECT_DIR / "data" / "metadata.csv"

df.to_csv(output_path, index=False)

print("METADATA CREATED SUCCESSFULLY!")
print("Total videos:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nSaved at:", output_path)
