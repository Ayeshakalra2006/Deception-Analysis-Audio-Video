
import pandas as pd
import subprocess
import imageio_ffmpeg
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_DIR / "data" / "metadata.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FFmpeg location
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

# Read video metadata
df = pd.read_csv(METADATA_PATH)

success = 0
failed = 0

for index, row in df.iterrows():

    video_path = Path(row["video_path"])
    output_path = OUTPUT_DIR / f"{video_path.stem}.wav"

    command = [
        ffmpeg,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode == 0:
        success += 1
    else:
        failed += 1

    print(f"Processed: {index + 1}/{len(df)}")

print("\nAUDIO EXTRACTION COMPLETE!")
print("Successful:", success)
print("Failed:", failed)
print("Audio saved in:", OUTPUT_DIR)
