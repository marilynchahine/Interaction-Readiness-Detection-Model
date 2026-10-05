import json
import cv2
from pathlib import Path

# ==========================
# Configuration
# ==========================
FRAMES_ROOT_DIR = Path(r"D:\EngagementDetection\data\frames\AVIDAR")
JSON_ROOT_DIR = Path(r"D:\EngagementDetection\data\JSON_hierarchical\AVIDAR")
OUTPUT_ROOT_DIR = Path(r"D:\EngagementDetection\data\cropped_frames\AVIDAR")

FRAME_EXT = ".jpg"
MARGIN = 0.0


# ==========================
# Helper functions
# ==========================
def expand_and_clip_bbox(bbox, img_w, img_h, margin=0.0):
    x, y, w, h = bbox

    if margin > 0:
        dx = w * margin
        dy = h * margin
        x -= dx
        y -= dy
        w += 2 * dx
        h += 2 * dy

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(img_w, int(round(x + w)))
    y2 = min(img_h, int(round(y + h)))

    return x1, y1, x2, y2


def process_video(frames_dir, json_path, output_root):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_id = data.get("video_id", json_path.stem)

    video_output_dir = output_root / video_id
    video_output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for person_id, annotations in data.items():

        if person_id == "video_id":
            continue

        if not isinstance(annotations, list):
            continue

        person_output_dir = video_output_dir / person_id
        person_output_dir.mkdir(parents=True, exist_ok=True)

        annotations = sorted(annotations, key=lambda x: x["frame_id"])

        saved_count = 0

        for ann in annotations:
            frame_id = ann["frame_id"]
            bbox = ann["bbox"]
            label = ann.get("label", "unknown")

            frame_path = frames_dir / f"{frame_id:06d}{FRAME_EXT}"

            if not frame_path.exists():
                print(f"Warning: missing frame {frame_path}")
                continue

            frame = cv2.imread(str(frame_path))

            if frame is None:
                print(f"Warning: could not read frame {frame_path}")
                continue

            img_h, img_w = frame.shape[:2]

            x1, y1, x2, y2 = expand_and_clip_bbox(
                bbox,
                img_w,
                img_h,
                margin=MARGIN
            )

            if x2 <= x1 or y2 <= y1:
                print(
                    f"Warning: invalid bbox in {json_path.name}, "
                    f"{person_id}, frame {frame_id}"
                )
                continue

            crop = frame[y1:y2, x1:x2]

            crop_name = f"{frame_id:06d}_{label}.jpg"
            crop_path = person_output_dir / crop_name

            cv2.imwrite(str(crop_path), crop)

            saved_count += 1
            total_saved += 1

        print(f"  {person_id}: saved {saved_count} crops")

    print(f"Finished {video_id}: saved {total_saved} crops\n")


# ==========================
# Main batch processing
# ==========================
OUTPUT_ROOT_DIR.mkdir(parents=True, exist_ok=True)

video_dirs = sorted(
    [p for p in FRAMES_ROOT_DIR.iterdir() if p.is_dir()],
    key=lambda p: p.name.lower()
)

for frames_dir in video_dirs:
    video_name = frames_dir.name
    json_path = JSON_ROOT_DIR / f"{video_name}.json"

    if not json_path.exists():
        print(f"Skipping {video_name}: no matching JSON file found")
        continue

    print(f"\nProcessing {video_name}")
    process_video(frames_dir, json_path, OUTPUT_ROOT_DIR)

print("All done.")