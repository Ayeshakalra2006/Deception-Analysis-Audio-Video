
from pathlib import Path

# Project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset folder
DATASET_DIR = PROJECT_DIR / "data" / "raw" / "Real-life_Deception_Detection_2016"

# Video folders
DECEPTIVE_DIR = DATASET_DIR / "Clips" / "Deceptive"
TRUTHFUL_DIR = DATASET_DIR / "Clips" / "Truthful"

print("PROJECT LOCATION:", PROJECT_DIR)
print("\nDATASET CHECK")
print("-" * 40)

if not DATASET_DIR.exists():
    print("Dataset folder nahi mila!")
    print("Expected location:", DATASET_DIR)

else:
    deceptive_videos = list(DECEPTIVE_DIR.glob("*.mp4"))
    truthful_videos = list(TRUTHFUL_DIR.glob("*.mp4"))

    print("Deceptive videos:", len(deceptive_videos))
    print("Truthful videos:", len(truthful_videos))
    print("Total videos:", len(deceptive_videos) + len(truthful_videos))

    print("\nDataset folders found successfully!")
