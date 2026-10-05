import cv2
from pathlib import Path

# ==========================
# Configuration
# ==========================
VIDEO_DIR = Path(r"D:\EngagementDetection\data\videos\temp")
OUTPUT_DIR = Path(r"D:\EngagementDetection\Push_to_HF\frames\SSUP-HRI")


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".MP4"}

# ==========================
# Create output directory
# ==========================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Process all videos
# ==========================
videos = sorted(
    [f for f in VIDEO_DIR.iterdir()
     if f.is_file() and f.suffix in VIDEO_EXTENSIONS],
    key=lambda x: x.name.lower()
)

for video_path in videos:

    print(f"\nProcessing {video_path.name}")

    # Create folder with same name as video
    video_output_dir = OUTPUT_DIR / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = f"{frame_idx:06d}.jpg"
        frame_path = video_output_dir / frame_filename

        cv2.imwrite(str(frame_path), frame)

        frame_idx += 1

    cap.release()

    print(f"Extracted {frame_idx} frames to {video_output_dir.name}")

print("\nDone.")