import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import cv2


json_dir = Path(
    r"D:\EngagementDetection\data\JSON_hierarchical\SSUP-HRI\AstorPlace_final"
)

output_dir = Path(
    r"D:\EngagementDetection\data\CVAT_xml_processed\SSUP-HRI\AstorPlace_final"
)

# Change this to the directory containing the source videos.
# The video filenames must match video_id, for example:
#
# video_id: "Sep-28-astor-place-24"
# video:    "Sep-28-astor-place-24.mp4"
video_dir = Path(
    r"D:\EngagementDetection\data\videos\SSUP-HRI\AstorPlace"
)

LABEL_NAME = "person"
ATTRIBUTE_NAME = "interaction_readiness"

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
)


def prettify_xml(root):
    rough = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def add_box(
    track_el,
    frame_id,
    bbox,
    label=None,
    outside=False,
):
    """
    Add one CVAT box.

    bbox must use:
        [x, y, width, height]

    When outside=True, the coordinates are retained from the previous
    visible box, but CVAT interprets the box as no longer visible.
    """
    x, y, w, h = bbox

    box_el = ET.SubElement(
        track_el,
        "box",
        {
            "frame": str(frame_id),
            "outside": "1" if outside else "0",
            "occluded": "0",
            "keyframe": "1",
            "xtl": f"{float(x):.2f}",
            "ytl": f"{float(y):.2f}",
            "xbr": f"{float(x + w):.2f}",
            "ybr": f"{float(y + h):.2f}",
            "z_order": "0",
        },
    )

    # Outside boxes only mark when a track disappears.
    if not outside and label is not None:
        attr_el = ET.SubElement(
            box_el,
            "attribute",
            {"name": ATTRIBUTE_NAME},
        )
        attr_el.text = str(label)


def find_video(video_id, video_dir):
    """
    Find the video corresponding to video_id.
    """
    for extension in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{extension}"

        if candidate.exists():
            return candidate

    # Case-insensitive fallback for Windows or unusual filenames.
    for path in video_dir.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and path.stem.lower() == str(video_id).lower()
        ):
            return path

    raise FileNotFoundError(
        f"Could not find a video for '{video_id}' in:\n"
        f"{video_dir}"
    )


def get_video_frame_count(video_path):
    """
    Return the total number of frames in a video.

    For a zero-indexed CVAT task:
        total_frames = 600
        valid frame IDs = 0 to 599
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    try:
        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        capture.release()

    if frame_count <= 0:
        raise RuntimeError(
            f"Could not determine frame count for: {video_path}"
        )

    return frame_count


def validate_bbox(json_path, person_key, frame_id, bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(
            f"{json_path.name}: invalid bbox for {person_key} "
            f"on frame {frame_id}: {bbox}"
        )

    try:
        x, y, width, height = map(float, bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{json_path.name}: bbox for {person_key} on frame "
            f"{frame_id} must contain four numbers: {bbox}"
        ) from exc

    if width <= 0 or height <= 0:
        raise ValueError(
            f"{json_path.name}: bbox for {person_key} on frame "
            f"{frame_id} has non-positive dimensions: {bbox}"
        )

    return [x, y, width, height]


def convert_json_to_cvat(
    json_path,
    output_dir,
    video_dir,
):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "video_id" not in data:
        raise ValueError(
            f"{json_path.name}: missing required 'video_id' field."
        )

    video_id = str(data["video_id"])

    video_path = find_video(
        video_id=video_id,
        video_dir=video_dir,
    )

    total_frames = get_video_frame_count(video_path)
    last_valid_frame = total_frames - 1

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    track_count = 0
    visible_box_count = 0
    outside_box_count = 0

    for key, annotations in data.items():
        if not key.startswith("person_"):
            continue

        if not isinstance(annotations, list):
            raise ValueError(
                f"{json_path.name}: {key} must contain a list "
                f"of annotations."
            )

        # An empty person list is valid and creates no CVAT track.
        if not annotations:
            continue

        person_id = key.replace("person_", "", 1)

        annotations = sorted(
            annotations,
            key=lambda annotation: int(
                annotation["frame_id"]
            ),
        )

        track_el = ET.SubElement(
            root,
            "track",
            {
                "id": str(person_id),
                "label": LABEL_NAME,
                "source": "manual",
            },
        )

        track_count += 1

        previous_frame = None
        previous_bbox = None
        seen_frames = set()

        for annotation in annotations:
            if "frame_id" not in annotation:
                raise ValueError(
                    f"{json_path.name}: missing frame_id in {key}."
                )

            frame_id = int(annotation["frame_id"])

            if frame_id < 0 or frame_id >= total_frames:
                raise ValueError(
                    f"{json_path.name}: {key} contains frame "
                    f"{frame_id}, but valid frames for the video are "
                    f"0 to {last_valid_frame}."
                )

            if frame_id in seen_frames:
                raise ValueError(
                    f"{json_path.name}: duplicate annotation for "
                    f"{key} on frame {frame_id}."
                )

            seen_frames.add(frame_id)

            if "bbox" not in annotation:
                raise ValueError(
                    f"{json_path.name}: missing bbox for {key} "
                    f"on frame {frame_id}."
                )

            if "label" not in annotation:
                raise ValueError(
                    f"{json_path.name}: missing label for {key} "
                    f"on frame {frame_id}."
                )

            bbox = validate_bbox(
                json_path=json_path,
                person_key=key,
                frame_id=frame_id,
                bbox=annotation["bbox"],
            )

            label = annotation["label"]

            # A gap means the person disappears after the previous
            # visible frame and later reappears.
            #
            # Example:
            # visible on frame 100
            # next annotation on frame 120
            #
            # Add outside="1" on frame 101.
            if (
                previous_frame is not None
                and frame_id > previous_frame + 1
            ):
                add_box(
                    track_el=track_el,
                    frame_id=previous_frame + 1,
                    bbox=previous_bbox,
                    outside=True,
                )
                outside_box_count += 1

            add_box(
                track_el=track_el,
                frame_id=frame_id,
                bbox=bbox,
                label=label,
                outside=False,
            )
            visible_box_count += 1

            previous_frame = frame_id
            previous_bbox = bbox

        # Crucial correction:
        #
        # Use the actual final video frame—not the highest annotated
        # frame anywhere in the JSON.
        #
        # If this person's last visible frame is not the final frame of
        # the video, explicitly mark the track outside on the next frame.
        if previous_frame < last_valid_frame:
            add_box(
                track_el=track_el,
                frame_id=previous_frame + 1,
                bbox=previous_bbox,
                outside=True,
            )
            outside_box_count += 1

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / f"{video_id}.xml"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(prettify_xml(root))

    print(
        f"Saved: {output_path} "
        f"| video frames: {total_frames} "
        f"| valid frame IDs: 0-{last_valid_frame} "
        f"| tracks: {track_count} "
        f"| visible boxes: {visible_box_count} "
        f"| outside boxes: {outside_box_count}"
    )


def main():

    if not json_dir.exists():
        raise FileNotFoundError(
            f"JSON directory does not exist:\n{json_dir}"
        )

    if not video_dir.exists():
        raise FileNotFoundError(
            f"Video directory does not exist:\n{video_dir}"
        )

    json_paths = sorted(
        json_dir.glob("*.json"),
        key=lambda path: path.name.lower(),
    )

    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files were found in:\n{json_dir}"
        )

    successful = 0
    failed = 0

    for json_path in json_paths:
        try:
            convert_json_to_cvat(
                json_path=json_path,
                output_dir=output_dir,
                video_dir=video_dir,
            )
            successful += 1

        except Exception as exc:
            failed += 1
            print()
            print(f"Failed: {json_path.name}")
            print(f"Reason: {exc}")
            print()

    print()
    print("Conversion complete.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()