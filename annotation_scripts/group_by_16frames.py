"""
Convert balanced_temporal_dataset.csv into one JSON file
per 16-frame sliding temporal window.

Input CSV format:
    source_dataset
    video_name
    frame_number
    frame_label
    person_id
    person_label
    bbox
    frame_path
    split

The CSV contains one row per person.

Bounding boxes in the CSV are expected as:
    [x, y, width, height]

Bounding boxes in the output JSON are converted to:
    [x1, y1, x2, y2]

Temporal grouping:
    group 1 -> frames 1-16
    group 2 -> frames 2-17
    group 3 -> frames 3-18
    ...

Groups are created independently for each video.

Output example:

{
    "video_name": "ex_video_name",
    "frame_group": 1,
    "split": "Train",
    "frames": [
        {
            "frame_number": 1,
            "frame_path": "...",
            "people": [
                {
                    "person_ID": 1,
                    "bbox": [x1, y1, x2, y2],
                    "label": "not_interaction_ready"
                }
            ]
        },
        ...
    ]
}
"""

from pathlib import Path
import ast
import json
import re

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = Path(
    r"/home/marilyn/Downloads/data/balanced_temporal_dataset.csv"
)

OUTPUT_DIR = Path(
    r"/home/marilyn/Downloads/data/JSON_temporal"
)

# Number of frames given to Qwen for each sample.
WINDOW_SIZE = 16

# Sliding-window stride:
#
# 1 means:
#   1-16
#   2-17
#   3-18
#   ...
#
# Change to 2, 4, etc. later if you want less overlap.
STRIDE = 1


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = {
    "source_dataset",
    "video_name",
    "frame_number",
    "person_id",
    "person_label",
    "bbox",
    "frame_path",
    "split",
}

VALID_LABELS = {
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
}


# ============================================================
# HELPERS
# ============================================================

def parse_bbox(value):
    """
    Parse a bbox stored in the CSV.

    Expected CSV format:
        [x, y, width, height]
    """

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except Exception as exc:
            raise ValueError(
                f"Could not parse bbox: {value}"
            ) from exc

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Bbox must be a list or tuple. Got: {value}"
        )

    if len(value) != 4:
        raise ValueError(
            f"Bbox must contain exactly 4 values. Got: {value}"
        )

    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Bbox contains non-numeric values: {value}"
        ) from exc


def bbox_xywh_to_xyxy(value):
    """
    Convert:

        [x, y, width, height]

    to:

        [x1, y1, x2, y2]
    """

    x, y, width, height = parse_bbox(value)

    return [
        round(x, 2),
        round(y, 2),
        round(x + width, 2),
        round(y + height, 2),
    ]


def normalize_person_id(value):
    """
    Preserve the person IDs already assigned by the temporal
    balancing script.

    Numeric IDs are written to JSON as integers.
    """

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

    except (TypeError, ValueError):
        pass

    return str(value).strip()


def person_sort_key(value):
    """
    Sort person IDs numerically whenever possible.
    """

    try:
        return 0, int(float(value))
    except (TypeError, ValueError):
        return 1, str(value)


def frame_sort_key(value):
    """
    Sort frame numbers numerically whenever possible.
    """

    try:
        return 0, int(float(value))
    except (TypeError, ValueError):
        return 1, str(value)


def normalize_frame_number(value):
    """
    Write normal numeric frame numbers as integers in JSON.
    """

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

    except (TypeError, ValueError):
        pass

    return str(value)


def normalize_split(value):
    """
    Normalize common split spellings to:

        Train
        Val
        Test
    """

    split = str(value).strip()

    mapping = {
        "train": "Train",
        "training": "Train",
        "val": "Val",
        "validation": "Val",
        "valid": "Val",
        "test": "Test",
        "testing": "Test",
    }

    return mapping.get(
        split.lower(),
        split,
    )


def safe_filename(value):
    """
    Make a string safe to use as part of a filename.
    """

    value = str(value)

    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        value,
    )

    value = value.strip(" .")

    return value or "video"


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(csv_path):
    """
    Load and validate the temporal CSV.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV does not exist:\n{csv_path}"
        )

    print(f"Loading dataset:\n{csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "The CSV is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    columns_that_cannot_be_missing = [
        "source_dataset",
        "video_name",
        "frame_number",
        "person_id",
        "person_label",
        "bbox",
        "frame_path",
        "split",
    ]

    for column in columns_that_cannot_be_missing:

        if df[column].isna().any():

            bad_rows = df[
                df[column].isna()
            ].index.tolist()[:10]

            raise ValueError(
                f"Column {column!r} contains missing values. "
                f"Example row indices: {bad_rows}"
            )

    # --------------------------------------------------------
    # Clean string columns
    # --------------------------------------------------------

    for column in [
        "source_dataset",
        "video_name",
        "person_label",
        "frame_path",
        "split",
    ]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    labels = set(
        df["person_label"].unique()
    )

    invalid_labels = (
        labels - VALID_LABELS
    )

    if invalid_labels:
        raise ValueError(
            "Unsupported person labels found:\n"
            + "\n".join(
                f"  - {label}"
                for label in sorted(invalid_labels)
            )
        )

    # --------------------------------------------------------
    # Validate bounding boxes now so errors are caught early.
    # --------------------------------------------------------

    for index, bbox in df["bbox"].items():

        try:
            parse_bbox(bbox)

        except ValueError as exc:

            raise ValueError(
                f"Invalid bbox at CSV row {index}: {bbox}"
            ) from exc

    print(
        f"Loaded {len(df):,} person rows."
    )

    return df


# ============================================================
# FRAME CREATION
# ============================================================

def build_frames_for_video(video_df):
    """
    Convert all person rows belonging to one video into
    frame-level dictionaries.

    Example input:

        frame 10 | person 1 | ...
        frame 10 | person 2 | ...
        frame 11 | person 1 | ...

    becomes:

        [
            {
                frame_number: 10,
                frame_path: ...,
                people: [...]
            },
            {
                frame_number: 11,
                frame_path: ...,
                people: [...]
            }
        ]
    """

    frames = []

    frame_groups = video_df.groupby(
        "frame_number",
        sort=False,
        dropna=False,
    )

    # Sort the unique frames chronologically.
    sorted_frame_numbers = sorted(
        frame_groups.groups.keys(),
        key=frame_sort_key,
    )

    for frame_number in sorted_frame_numbers:

        frame_df = frame_groups.get_group(
            frame_number
        ).copy()

        # ----------------------------------------------------
        # Verify frame path
        # ----------------------------------------------------

        frame_paths = (
            frame_df["frame_path"]
            .astype(str)
            .str.strip()
            .unique()
        )

        if len(frame_paths) != 1:
            raise ValueError(
                f"Video {frame_df['video_name'].iloc[0]!r}, "
                f"frame {frame_number}: expected exactly one "
                f"frame_path but found {len(frame_paths)}."
            )

        frame_path = frame_paths[0]

        # ----------------------------------------------------
        # Verify split
        # ----------------------------------------------------

        frame_splits = (
            frame_df["split"]
            .astype(str)
            .str.strip()
            .unique()
        )

        if len(frame_splits) != 1:
            raise ValueError(
                f"Video {frame_df['video_name'].iloc[0]!r}, "
                f"frame {frame_number}: multiple split values "
                f"found: {frame_splits}"
            )

        # ----------------------------------------------------
        # Sort people by the already-assigned person_id.
        #
        # IMPORTANT:
        # We DO NOT enumerate them 1...N here.
        #
        # person_id 1 must remain the same person throughout
        # the entire video.
        # ----------------------------------------------------

        frame_df["_person_sort"] = (
            frame_df["person_id"]
            .apply(person_sort_key)
        )

        frame_df = frame_df.sort_values(
            "_person_sort"
        )

        people = []

        seen_person_ids = set()

        for _, row in frame_df.iterrows():

            person_id = normalize_person_id(
                row["person_id"]
            )

            if person_id in seen_person_ids:
                raise ValueError(
                    f"Duplicate person_ID {person_id} "
                    f"in video {row['video_name']!r}, "
                    f"frame {frame_number}."
                )

            seen_person_ids.add(
                person_id
            )

            label = str(
                row["person_label"]
            ).strip()

            people.append(
                {
                    "person_ID": person_id,
                    "bbox": bbox_xywh_to_xyxy(
                        row["bbox"]
                    ),
                    "label": label,
                }
            )

        frames.append(
            {
                "frame_number":
                    normalize_frame_number(
                        frame_number
                    ),
                "frame_path":
                    frame_path,
                "people":
                    people,
            }
        )

    return frames


# ============================================================
# JSON GENERATION
# ============================================================

def create_video_groups(
    video_df,
    output_dir,
):
    """
    Create sliding WINDOW_SIZE-frame JSON groups for one video.
    """

    source_dataset = str(
        video_df["source_dataset"].iloc[0]
    ).strip()

    video_name = str(
        video_df["video_name"].iloc[0]
    ).strip()

    # --------------------------------------------------------
    # A complete video must belong to exactly one split.
    # --------------------------------------------------------

    splits = (
        video_df["split"]
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(splits) != 1:
        raise ValueError(
            f"Video {video_name!r} appears in multiple splits: "
            f"{splits.tolist()}. "
            f"Videos must be split before temporal window creation."
        )

    split = normalize_split(
        splits[0]
    )

    frames = build_frames_for_video(
        video_df
    )

    num_frames = len(frames)

    if num_frames < WINDOW_SIZE:

        print(
            f"Skipping {source_dataset}/{video_name}: "
            f"only {num_frames} frames "
            f"(need at least {WINDOW_SIZE})."
        )

        return 0

    # --------------------------------------------------------
    # Keep outputs separated by split.
    #
    # Example:
    #
    # temporal_frame_groups/
    #     Train/
    #     Val/
    #     Test/
    # --------------------------------------------------------

    split_output_dir = (
        output_dir / split
    )

    split_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    number_of_groups = 0

    # --------------------------------------------------------
    # Sliding windows
    # --------------------------------------------------------

    for start_idx in range(
        0,
        num_frames - WINDOW_SIZE + 1,
        STRIDE,
    ):

        end_idx = (
            start_idx + WINDOW_SIZE
        )

        window_frames = frames[
            start_idx:end_idx
        ]

        number_of_groups += 1

        group_json = {
            "video_name":
                video_name,

            "frame_group":
                number_of_groups,

            "split":
                split,

            "frames":
                window_frames,
        }

        # ----------------------------------------------------
        # Filename
        # ----------------------------------------------------

        safe_dataset = safe_filename(
            source_dataset
        )

        safe_video = safe_filename(
            video_name
        )

        filename = (
            f"{safe_dataset}"
            f"__{safe_video}"
            f"__group_{number_of_groups:06d}.json"
        )

        output_path = (
            split_output_dir / filename
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                group_json,
                file,
                indent=2,
                ensure_ascii=False,
            )

    print(
        f"{source_dataset}/{video_name}: "
        f"{num_frames} frames -> "
        f"{number_of_groups} groups"
    )

    return number_of_groups


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )
    print(
        "TEMPORAL CSV -> 16-FRAME JSON GROUPS"
    )
    print(
        "========================================\n"
    )

    print(
        f"Window size: {WINDOW_SIZE}"
    )

    print(
        f"Stride:      {STRIDE}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset(
        INPUT_CSV
    )

    # --------------------------------------------------------
    # Validate source_dataset within each video identity.
    #
    # We use both source_dataset and video_name as the video key
    # in case two datasets contain videos with the same name.
    # --------------------------------------------------------

    video_columns = [
        "source_dataset",
        "video_name",
    ]

    grouped_videos = df.groupby(
        video_columns,
        sort=True,
        dropna=False,
    )

    total_videos = (
        grouped_videos.ngroups
    )

    total_groups = 0

    groups_by_split = {
        "Train": 0,
        "Val": 0,
        "Test": 0,
    }

    processed_videos = 0

    for (
        source_dataset,
        video_name,
    ), video_df in grouped_videos:

        num_groups = create_video_groups(
            video_df=video_df,
            output_dir=OUTPUT_DIR,
        )

        processed_videos += 1
        total_groups += num_groups

        if num_groups > 0:

            split_values = (
                video_df["split"]
                .astype(str)
                .str.strip()
                .unique()
            )

            split = normalize_split(
                split_values[0]
            )

            groups_by_split.setdefault(
                split,
                0,
            )

            groups_by_split[split] += (
                num_groups
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print(
        "DONE"
    )
    print(
        "========================================"
    )

    print(
        f"Videos processed: {processed_videos:,} "
        f"/ {total_videos:,}"
    )

    print(
        f"JSON groups:      {total_groups:,}"
    )

    print(
        "\nGroups by split:"
    )

    for split, count in sorted(
        groups_by_split.items()
    ):
        print(
            f"  {split:<10} {count:,}"
        )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
