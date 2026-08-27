import json
import xml.etree.ElementTree as ET
from pathlib import Path


# Define the input and output directories here
XML_DIR = Path(r"D:\\EngagementDetection\\data\\CVAT_xml_not_processed\\SSUP-HRI\\AstorPlace")
OUTPUT_DIR = Path(r"D:\\EngagementDetection\\data\\JSON_hierarchical\\SSUP-HRI\\AstorPlace")


DEFAULT_LABEL = "not_interaction_ready"
ATTRIBUTE_NAME = "interaction_readiness"


def get_video_id(root, xml_path):
    # Prefer original video filename from CVAT metadata if available
    task_name = root.findtext(".//meta/task/name")
    if task_name:
        return Path(task_name).stem

    return xml_path.stem


def get_box_label(box):
    for attr in box.findall("attribute"):
        if attr.attrib.get("name") == ATTRIBUTE_NAME:
            if attr.text:
                return attr.text.strip()

    return DEFAULT_LABEL


def convert_xml(xml_path: Path, output_dir: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    video_id = get_video_id(root, xml_path)

    output = {
        "video_id": video_id
    }

    for track in root.findall("track"):
        if track.attrib.get("label") != "person":
            continue

        person_id = track.attrib["id"]
        person_key = f"person_{person_id}"

        output[person_key] = []

        for box in track.findall("box"):
            if box.attrib.get("outside") == "1":
                continue

            frame_id = int(box.attrib["frame"])

            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            bbox = [
                round(xtl, 2),
                round(ytl, 2),
                round(xbr - xtl, 2),
                round(ybr - ytl, 2),
            ]

            output[person_key].append({
                "frame_id": frame_id,
                "bbox": bbox,
                "label": get_box_label(box),
            })

        output[person_key].sort(key=lambda x: x["frame_id"])

    output_path = output_dir / f"{video_id}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {output_path}")


def main():
    xml_dir = XML_DIR
    output_dir = OUTPUT_DIR

    for xml_path in sorted(xml_dir.glob("*.xml")):
        convert_xml(xml_path, output_dir)


if __name__ == "__main__":
    main()