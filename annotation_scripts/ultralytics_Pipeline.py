from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.dom import minidom

import torch

print("GPU Available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    device = 0
else:
    device = "cpu"

INPUT_DIR = Path("D:\\EngagementDetection\\data\\videos\\SSUP-HRI\\AstorPlace")
OUTPUT_DIR = Path("D:\\EngagementDetection\\data\\CVAT_xml_not_processed\\SSUP-HRI\\AstorPlace")
MODEL_PATH = "yolo11x.pt"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def save_cvat_xml(tracks, output_xml):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    for track_id, boxes in sorted(tracks.items()):
        track_el = ET.SubElement(
            root,
            "track",
            {
                "id": str(track_id),
                "label": "person",
                "source": "manual",
            },
        )

        for frame, x1, y1, x2, y2 in boxes:
            ET.SubElement(
                track_el,
                "box",
                {
                    "frame": str(frame),
                    "outside": "0",
                    "occluded": "0",
                    "keyframe": "1",
                    "xtl": f"{x1:.2f}",
                    "ytl": f"{y1:.2f}",
                    "xbr": f"{x2:.2f}",
                    "ybr": f"{y2:.2f}",
                    "z_order": "0",
                },
            )

    pretty_xml = minidom.parseString(
        ET.tostring(root, encoding="utf-8")
    ).toprettyxml(indent="  ")

    output_xml.write_text(pretty_xml, encoding="utf-8")


def process_video(model, video_path, output_xml):
    print(f"Processing: {video_path.name}")

    tracks = defaultdict(list)

    results = model.track(
        source=str(video_path),
        tracker="botsort.yaml",
        classes=[0],        # person only
        persist=True,
        stream=True,
        verbose=False,
        device = device,
    )

    for frame_idx, result in enumerate(results):  # CVAT frames start at 0
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            tracks[track_id].append((frame_idx, x1, y1, x2, y2))

    save_cvat_xml(tracks, output_xml)

    print(f"Saved: {output_xml}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_files = [
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        raise FileNotFoundError(f"No video files found in {INPUT_DIR}")

    model = YOLO(MODEL_PATH)
    model.to(device)

    for video_path in video_files:
        output_xml = OUTPUT_DIR / f"{video_path.stem}_cvat.xml"
        process_video(model, video_path, output_xml)


if __name__ == "__main__":
    main()