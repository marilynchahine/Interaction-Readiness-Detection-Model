from pathlib import Path
import json
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

JSON_DIR = Path(
    r"D:\EngagementDetection\data\JSON_hierarchical\SSUP-HRI\AstorPlace_final"
)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_interaction_done_frames(json_dir):
    total_done_only_frames = 0
    total_done_mixed_frames = 0

    per_video_results = []

    json_paths = sorted(json_dir.glob("*.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files found in:\n{json_dir}"
        )

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_id = str(data.get("video_id", json_path.stem))

        # frame_id -> set of labels appearing in that frame
        labels_by_frame = defaultdict(set)

        for person_id, annotations in data.items():

            if person_id == "video_id":
                continue

            if not isinstance(annotations, list):
                continue

            for annotation in annotations:

                if not isinstance(annotation, dict):
                    continue

                frame_id = annotation.get("frame_id")
                label = annotation.get("label")

                if frame_id is None or label is None:
                    continue

                label = str(label).strip()

                labels_by_frame[str(frame_id)].add(label)

        done_only_count = 0
        done_mixed_count = 0

        for frame_id, labels in labels_by_frame.items():

            # Case 1:
            # Every bounding box/person in the frame is interaction_done.
            if labels == {"interaction_done"}:
                done_only_count += 1

            # Case 2:
            # interaction_done exists together with at least one
            # different label.
            elif (
                "interaction_done" in labels
                and len(labels) > 1
            ):
                done_mixed_count += 1

        total_done_only_frames += done_only_count
        total_done_mixed_frames += done_mixed_count

        per_video_results.append(
            {
                "video_id": video_id,
                "done_only_frames": done_only_count,
                "done_with_other_label_frames": done_mixed_count,
            }
        )

    return (
        total_done_only_frames,
        total_done_mixed_frames,
        per_video_results,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    (
        total_done_only,
        total_done_mixed,
        per_video_results,
    ) = analyze_interaction_done_frames(JSON_DIR)

    print("\n========== INTERACTION_DONE FRAME STATISTICS ==========")

    print(
        f"\nFrames containing ONLY interaction_done: "
        f"{total_done_only}"
    )

    print(
        f"Frames containing interaction_done + another label: "
        f"{total_done_mixed}"
    )

    print(
        f"Total frames containing interaction_done: "
        f"{total_done_only + total_done_mixed}"
    )

    print("\nPer-video statistics:")

    for result in per_video_results:
        print(f"\nVideo: {result['video_id']}")
        print(
            f"  interaction_done only: "
            f"{result['done_only_frames']}"
        )
        print(
            f"  interaction_done + other label: "
            f"{result['done_with_other_label_frames']}"
        )

    print("\n=======================================================")