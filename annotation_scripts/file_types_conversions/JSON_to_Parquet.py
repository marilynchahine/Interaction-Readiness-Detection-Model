import json
from pathlib import Path

import pandas as pd

from datasets import Dataset

# =============================================================================
# Dataset Format (one row per person per frame)
#
# Columns:
#   source_dataset : str
#       Name of the original dataset (e.g., AVIDAR, AVIDAR, UE-HRI, AVIDAR).
#
#   video_id : str
#       Unique identifier of the video.
#
#   person_id : str
#       Identifier of the tracked person within the video.
#
#   frame_id : int
#       Frame number in the video.
#
#   frame_path : str
#       Relative path to the full frame image from the dataset root.
#       Example:
#           data/frames/AVIDAR_001/000042.jpg
#
#   crop_path : str
#       Relative path to the cropped image of the person from the dataset root.
#       Example:
#           data/crops/AVIDAR_001/person_3/000042.jpg
#
#   bbox : list[float]
#       Bounding box of the person in the format:
#           [x, y, width, height]
#
#   label : str
#       Interaction readiness label for the person in that frame.
#       Example:
#           interaction_ready
#           not_interaction_ready
#
# Each row represents a single person's annotation in a single frame.
# =============================================================================


# ============================================================
# Configuration
# ============================================================

JSON_DIR = Path("D:\EngagementDetection\data\JSON_hierarchical\SSUP-HRI\AstorPlace_final")
OUTPUT_PARQUET = Path("D:\\EngagementDetection\\data\\Parquet\\SSUP-HRI.parquet")

SOURCE_DATASET = "SSUP-HRI"
FRAME_EXTENSION = ".jpg"

# ============================================================


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_annotation(json_path):
    data = load_json(json_path)

    video_id = data.get("video_id", json_path.stem)
    rows = []

    for person_id, annotations in data.items():
        if person_id == "video_id":
            continue

        if not isinstance(annotations, list):
            continue

        for ann in annotations:
            frame_id = int(ann["frame_id"])
            frame_filename = f"{frame_id:06d}{FRAME_EXTENSION}"

            rows.append(
                {
                    "source_dataset": SOURCE_DATASET,
                    "video_id": video_id,
                    "person_id": person_id,
                    "frame_id": frame_id,
                    "frame_path": f"D:\\EngagementDetection\\Push_to_HF\\frames\\{SOURCE_DATASET}\\{video_id}\\{frame_filename}",
                    "bbox": [float(x) for x in ann["bbox"]],
                    "label": ann["label"],
                }
            )

    return rows


def convert_folder_to_parquet():
    json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {JSON_DIR}")

    all_rows = []

    for json_path in json_files:
        print(f"Processing {json_path.name}")
        all_rows.extend(flatten_annotation(json_path))

    df = pd.DataFrame(
        all_rows,
        columns=[
            "source_dataset",
            "video_id",
            "person_id",
            "frame_id",
            "frame_path",
            "bbox",
            "label",
        ],
    )

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print()
    print(f"Saved {len(df)} annotations to {OUTPUT_PARQUET}")

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nNumber of videos:", df["video_id"].nunique())
    print("Number of person tracks:", df[["video_id", "person_id"]].drop_duplicates().shape[0])


    verify_parquet(df)


##############################################
#               Sanity Checks                #
##############################################


def verify_parquet(df):
    print("\n" + "=" * 70)
    print("Verifying Parquet file...")
    print("=" * 70)

    # ---------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------
    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------
    print("\nMissing values:")
    missing = df.isnull().sum()

    if missing.sum() == 0:
        print("✓ No missing values.")
    else:
        print(missing)

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------
    duplicates = df.duplicated(
        subset=[
            "source_dataset",
            "video_id",
            "person_id",
            "frame_id",
            "frame_path",
            "label",
            ]
        ).sum()

    if duplicates == 0:
        print("\n✓ No duplicate rows.")
    else:
        print(f"\n⚠ {duplicates} duplicate rows found.")

    # ---------------------------------------------------------
    # Dataset statistics
    # ---------------------------------------------------------
    print("\nNumber of videos:")
    print(df["video_id"].nunique())

    print("\nNumber of person tracks:")
    print(df[["video_id", "person_id"]].drop_duplicates().shape[0])

    print("\nNumber of annotations:")
    print(len(df))

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # ---------------------------------------------------------
    # Verify image paths
    # ---------------------------------------------------------
    missing_frames = 0

    for frame_path in df["frame_path"]:
        if not Path(frame_path).exists():
            missing_frames += 1

    print("\nImage path verification:")

    if missing_frames == 0:
        print("✓ All frame images exist.")
    else:
        print(f"⚠ Missing frame images: {missing_frames}")

    # ---------------------------------------------------------
    # Bounding box validation
    # ---------------------------------------------------------
    invalid_boxes = 0

    for bbox in df["bbox"]:
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            invalid_boxes += 1

    if invalid_boxes == 0:
        print("\n✓ All bounding boxes are valid.")
    else:
        print(f"\n⚠ Invalid bounding boxes: {invalid_boxes}")

    # ---------------------------------------------------------
    # Hugging Face compatibility
    # ---------------------------------------------------------
    try:
        dataset = Dataset.from_parquet(str(OUTPUT_PARQUET))

        print("\n✓ Successfully loaded with Hugging Face Datasets.")

        print("\nDetected schema:")
        print(dataset.features)

    except Exception as e:
        print("\n⚠ Hugging Face loading failed.")
        print(e)

    print("\n" + "=" * 70)
    print("Verification complete.")
    print("=" * 70)


if __name__ == "__main__":
    convert_folder_to_parquet()