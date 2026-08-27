import json
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


INPUT_DIR = Path("C:\\Users\\maril\\OneDrive\\Desktop\\GitHub\\InteractionReadiness\\IRDM\\data\\CVAT_files_bboxes\\JRDB_skeletons")
OUTPUT_DIR = Path("C:\\Users\\maril\\OneDrive\\Desktop\\GitHub\\InteractionReadiness\\IRDM\\data\\CVAT_files_bboxes\\JRDB_bboxes")

LABEL_NAME = "person"
VISIBILITY_THRESHOLD = 0  # keeps keypoints with v > 0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prettify_xml(elem):
    rough = ET.tostring(elem, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")


def keypoints_to_bbox(keypoints, img_w, img_h):
    xs, ys = [], []

    for i in range(0, len(keypoints), 3):
        x = keypoints[i]
        y = keypoints[i + 1]
        v = keypoints[i + 2]

        if v > VISIBILITY_THRESHOLD:
            xs.append(x)
            ys.append(y)

    if not xs:
        return None

    xtl = max(0, min(xs))
    ytl = max(0, min(ys))
    xbr = min(img_w, max(xs))
    ybr = min(img_h, max(ys))

    if xbr <= xtl or ybr <= ytl:
        return None

    return xtl, ytl, xbr, ybr


def convert_json_to_cvat_xml(input_json, output_xml):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {img["id"]: img for img in data["images"]}
    tracks = {}

    for ann in data["annotations"]:
        image = images_by_id[ann["image_id"]]

        frame_number = int(Path(image["file_name"]).stem)
        img_w = image["width"]
        img_h = image["height"]

        bbox = keypoints_to_bbox(ann["keypoints"], img_w, img_h)
        if bbox is None:
            continue

        track_id = ann.get("track_id", ann["id"])

        tracks.setdefault(track_id, []).append({
            "frame": frame_number,
            "bbox": bbox
        })

    annotations = ET.Element("annotations")

    version = ET.SubElement(annotations, "version")
    version.text = "1.1"

    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")

    ET.SubElement(task, "name").text = input_json.stem
    ET.SubElement(task, "size").text = str(len(data["images"]))
    ET.SubElement(task, "mode").text = "interpolation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    ET.SubElement(task, "created").text = ""
    ET.SubElement(task, "updated").text = ""
    ET.SubElement(task, "start_frame").text = "0"
    ET.SubElement(task, "stop_frame").text = str(len(data["images"]) - 1)
    ET.SubElement(task, "frame_filter").text = ""

    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = LABEL_NAME
    ET.SubElement(label, "color").text = "#ff0000"
    ET.SubElement(label, "attributes")

    for idx, (track_id, boxes) in enumerate(sorted(tracks.items())):
        track = ET.SubElement(
            annotations,
            "track",
            {
                "id": str(idx),
                "label": LABEL_NAME,
                "source": "manual"
            }
        )

        boxes = sorted(boxes, key=lambda x: x["frame"])

        for item in boxes:
            xtl, ytl, xbr, ybr = item["bbox"]

            ET.SubElement(
                track,
                "box",
                {
                    "frame": str(item["frame"]),
                    "outside": "0",
                    "occluded": "0",
                    "keyframe": "1",
                    "xtl": f"{xtl:.2f}",
                    "ytl": f"{ytl:.2f}",
                    "xbr": f"{xbr:.2f}",
                    "ybr": f"{ybr:.2f}",
                    "z_order": "0"
                }
            )

    with open(output_xml, "w", encoding="utf-8") as f:
        f.write(prettify_xml(annotations))

    print(f"Converted {input_json.name} -> {output_xml.name} ({len(tracks)} tracks)")


for input_json in sorted(INPUT_DIR.glob("*.json")):
    output_xml = OUTPUT_DIR / f"{input_json.stem}.xml"
    convert_json_to_cvat_xml(input_json, output_xml)

print("Done.")