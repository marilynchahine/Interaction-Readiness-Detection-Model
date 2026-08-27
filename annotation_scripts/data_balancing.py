"""

Frame count rules:
- We don't use frames that include only 'interaction_done' bounding boxes, since they ressemble 'not_interaction_ready' too much.
- If a frame only contains bounding boxes labelled as 'not_interaction_ready", it will be counted as a 'not_interaction_ready' sample.
- If a frame contains at least one bounding box labelled as 'interaction_ready', it will be counted as a 'interaction_ready' sample, regardless of the presence of any 'not_interaction_ready' or 'interaction_ongoing' boxes.
- If a frame contains at least one bounding box labelled as 'interaction_ongoing' and no bounding boxes labelled as 'interaction_ready', it will be counted as an 'interaction_ongoing' sample, regardless of the presence of any 'not_interaction_ready' boxes.


Data balancing rules:
- All SSUP-HRI frames are included in the dataset since they provide most ofthe interaction_ready and interaction_ongoing labels.
- All AVIDAR 'interaction_ready' frames are included in the dataset since they provide a different type of interaction_ready frames.
- Remove frames that include only 'interaction_done' bounding boxes, since they ressemble 'not_interaction_ready' too much.
- Rename leftover 'interaction_done' bounding boxes to 'not_interaction_ready'.
- The currently included frames have more interaction_ready than not_interaction_ready samples, so we will include not_interaction_ready frames from the JRDB videos and the AVIDAR currently unused videos to balance the dataset.
- The number of not_interaction_ready frames to include from the JRDB and AVIDAR datasets corresponds to the difference between the number of interaction_ready frames and the number of not_interaction_ready frames in the currently included frames. Designate this number as N.
- To maximize video/scene diversity, we divide N equally across the total number of leftover videos in JRDB and AVIDAR. Designate this number as D, so D = N / (number of videos in JRDB and AVIDAR not_interaction_ready videos).
- For frame-based models: We randomly select D frames from each leftover video.
- For temporal models: We randomly select a sequence of D frames from each leftover video by selecting a start frame starting from the first frame and ending at the last frame minus D. The D frames are then selected as a continuous sequence starting from the randomly selected start frame.
- If the video has less than D frames, we will include all the frames in that video.


Additional steps:
- For frames with more than 5 bounding boxes, we will include the 5 bounding boxes with the highest area and discard the rest. This is to avoid including frames with too many bounding boxes, which may not be useful for training the model.


Output format:
- A CSV file that describes the balanced dataset for frame-based models with the following columns:
    - source_dataset
    - video_name
    - frame_number
    - frame_label
    - person_id
    - person_label
    - bbox
    - frame_path

- A dataset statistics CSV file with the following information for the frame-based models balanced dataset:
    - frame-level statistics:
        - total number of frames in the dataset
        - number of frames for each label
        - number of frames for each label per dataset (SSUP-HRI, JRDB, AVIDAR)
    - bounding box-level statistics:
        - total number of bounding boxes in the dataset
        - number of bounding boxes for each label
        - number of bounding boxes for each label per dataset (SSUP-HRI, JRDB, AVIDAR)
    - video-level statistics:
        - total number of videos in the dataset
        - number of videos for each dataset (SSUP-HRI, JRDB, AVIDAR)
        - number of frames for each label per video
        - number of bounding boxes for each label per video

- A CSV file that describes the balanced dataset for temporal models with the following columns:
    - source_dataset
    - video_name
    - frame_number
    - frame_label
    - person_id
    - person_label
    - bbox
    - frame_path

- A dataset statistics CSV file with the following information for the temporal models balanced dataset:
    - frame-level statistics:
        - total number of frames in the dataset
        - number of frames for each label
        - number of frames for each label per dataset (SSUP-HRI, JRDB, AVIDAR)
    - bounding box-level statistics:
        - total number of bounding boxes in the dataset
        - number of bounding boxes for each label
        - number of bounding boxes for each label per dataset (SSUP-HRI, JRDB, AVIDAR)
    - video-level statistics:
        - total number of videos in the dataset
        - number of videos for each dataset (SSUP-HRI, JRDB, AVIDAR)
        - number of frames for each label per video
        - number of bounding boxes for each label per video

"""

from pathlib import Path
import ast
import json
import math
import random
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Hierarchical JSON directories
# ------------------------------------------------------------

SSUP_JSON_DIR = Path(
    r"D:\EngagementDetection\data\JSON_hierarchical\SSUP-HRI\AstorPlace_final"
)

AVIDAR_JSON_DIR = Path(
    r"D:\EngagementDetection\data\JSON_hierarchical\AVIDAR"
)

JRDB_JSON_DIR = Path(
    r"D:\EngagementDetection\data\JSON_hierarchical\JRDB"
)


# ------------------------------------------------------------
# Frame/image root directories
#
# These are only used when frame_path is not already stored
# inside the JSON annotation.
#
# Change these to match your own frame directories.
# ------------------------------------------------------------

FRAME_ROOTS = {
    "SSUP-HRI": Path(
        r"D:\EngagementDetection\final\frames\SSUP-HRI"
    ),
    "AVIDAR": Path(
        r"D:\EngagementDetection\final\frames\AVIDAR"
    ),
    "JRDB": Path(
        r"D:\EngagementDetection\final\frames\JRDB"
    ),
}


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\EngagementDetection\final"
)

FRAME_DATASET_OUTPUT = (
    OUTPUT_DIR / "balanced_frame_based_dataset.csv"
)

FRAME_STATS_OUTPUT = (
    OUTPUT_DIR / "balanced_frame_based_dataset_statistics.csv"
)

TEMPORAL_DATASET_OUTPUT = (
    OUTPUT_DIR / "balanced_temporal_dataset.csv"
)

TEMPORAL_STATS_OUTPUT = (
    OUTPUT_DIR / "balanced_temporal_dataset_statistics.csv"
)


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

RANDOM_SEED = 42

# Maximum number of people/bounding boxes retained per frame.
MAX_BOXES_PER_FRAME = 5


# ============================================================
# CONSTANTS
# ============================================================

VALID_FINAL_LABELS = {
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
}

ALL_INPUT_LABELS = {
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
    "interaction_done",
}

LABEL_PRIORITY = [
    "interaction_ready",
    "interaction_ongoing",
    "not_interaction_ready",
]

OUTPUT_COLUMNS = [
    "source_dataset",
    "video_name",
    "frame_number",
    "frame_label",
    "person_id",
    "person_label",
    "bbox",
    "frame_path",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_label(label):
    """
    Clean an annotation label.
    """

    if label is None:
        return None

    label = str(label).strip()

    if not label:
        return None

    return label


def normalize_frame_number(frame_id):
    """
    Convert numeric frame IDs to integers when possible.

    Non-numeric frame IDs are preserved as strings.
    """

    if frame_id is None:
        return None

    try:
        return int(frame_id)
    except (TypeError, ValueError):
        return str(frame_id)


def frame_sort_key(frame_number):
    """
    Sorting key that handles both numeric and non-numeric IDs.
    """

    try:
        return 0, int(frame_number)
    except (TypeError, ValueError):
        return 1, str(frame_number)


def bbox_area(bbox):
    """
    Calculate bounding-box area.

    Expected bbox format:
        [x, y, width, height]

    Returns 0 if the bbox is malformed.
    """

    if bbox is None:
        return 0.0

    # Sometimes CSV/JSON processing may turn a bbox into a string.
    if isinstance(bbox, str):
        try:
            bbox = ast.literal_eval(bbox)
        except Exception:
            return 0.0

    if not isinstance(bbox, (list, tuple)):
        return 0.0

    if len(bbox) < 4:
        return 0.0

    try:
        width = float(bbox[2])
        height = float(bbox[3])

        return max(0.0, width) * max(0.0, height)

    except (TypeError, ValueError):
        return 0.0


# ============================================================
# FRAME PATH
# ============================================================

def resolve_frame_path(
    annotation,
    source_dataset,
    video_name,
    frame_number,
):
    """
    Determine the path of a video frame.

    Priority:

    1. Use an existing frame_path in the JSON.
    2. Use image_path if present.
    3. Try common filenames under FRAME_ROOTS.

    If your local frame directory uses another structure,
    modify this function.
    """

    # --------------------------------------------------------
    # Existing path in annotation
    # --------------------------------------------------------

    for key in [
        "frame_path",
        "image_path",
        "image",
    ]:
        value = annotation.get(key)

        if isinstance(value, str) and value.strip():
            return value

    # --------------------------------------------------------
    # Try constructing path
    # --------------------------------------------------------

    root = FRAME_ROOTS.get(source_dataset)

    if root is None:
        return ""

    video_dir = root / str(video_name)

    try:
        frame_int = int(frame_number)
    except (TypeError, ValueError):
        frame_int = None

    candidate_names = []

    if frame_int is not None:
        candidate_names.extend([
            f"{frame_int}.jpg",
            f"{frame_int}.jpeg",
            f"{frame_int}.png",
            f"{frame_int:06d}.jpg",
            f"{frame_int:06d}.jpeg",
            f"{frame_int:06d}.png",
            f"{frame_int:08d}.jpg",
            f"{frame_int:08d}.jpeg",
            f"{frame_int:08d}.png",
        ])
    else:
        candidate_names.extend([
            f"{frame_number}.jpg",
            f"{frame_number}.jpeg",
            f"{frame_number}.png",
        ])

    for filename in candidate_names:
        candidate = video_dir / filename

        if candidate.exists():
            return str(candidate)

    # No matching file found.
    return ""


# ============================================================
# FRAME LABEL AGGREGATION
# ============================================================

def determine_frame_label(person_labels):
    """
    Determine one label for an entire frame.

    Rules
    -----

    interaction_ready present
        -> interaction_ready

    else interaction_ongoing present
        -> interaction_ongoing

    else not_interaction_ready present
        -> not_interaction_ready

    else interaction_done only
        -> None / excluded

    This means:

    ready + ongoing
        -> ready

    ready + done
        -> ready

    ongoing + done
        -> ongoing

    not_ready + done
        -> not_ready

    done only
        -> excluded
    """

    labels = set(person_labels)

    if "interaction_ready" in labels:
        return "interaction_ready"

    if "interaction_ongoing" in labels:
        return "interaction_ongoing"

    if "not_interaction_ready" in labels:
        return "not_interaction_ready"

    if labels == {"interaction_done"}:
        return None

    return None


# ============================================================
# JSON LOADING
# ============================================================

def load_dataset_jsons(
    json_directory,
    source_dataset,
):
    """
    Load a hierarchical JSON dataset.

    Returns
    -------
    frames : dict

    Structure:

        {
            (source_dataset, video_name, frame_number): {
                "source_dataset": ...,
                "video_name": ...,
                "frame_number": ...,
                "frame_label": ...,
                "people": [...]
            }
        }
    """

    json_paths = sorted(json_directory.glob("*.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files found for {source_dataset} in:\n"
            f"{json_directory}"
        )

    frames = {}

    excluded_done_only = 0

    for json_path in json_paths:

        with open(
            json_path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        video_name = str(
            data.get("video_id", json_path.stem)
        )

        # ----------------------------------------------------
        # First collect people by frame
        # ----------------------------------------------------

        people_by_frame = defaultdict(list)

        for person_id, annotations in data.items():

            if person_id == "video_id":
                continue

            if not isinstance(annotations, list):
                continue

            for annotation in annotations:

                if not isinstance(annotation, dict):
                    continue

                frame_number = normalize_frame_number(
                    annotation.get("frame_id")
                )

                if frame_number is None:
                    continue

                label = normalize_label(
                    annotation.get("label")
                )

                if label not in ALL_INPUT_LABELS:
                    continue

                bbox = annotation.get("bbox")

                frame_path = resolve_frame_path(
                    annotation=annotation,
                    source_dataset=source_dataset,
                    video_name=video_name,
                    frame_number=frame_number,
                )

                people_by_frame[frame_number].append(
                    {
                        "person_id": str(person_id),
                        "person_label": label,
                        "bbox": bbox,
                        "frame_path": frame_path,
                    }
                )

        # ----------------------------------------------------
        # Assign one frame-level label
        # ----------------------------------------------------

        for frame_number, people in people_by_frame.items():

            person_labels = [
                person["person_label"]
                for person in people
            ]

            frame_label = determine_frame_label(
                person_labels
            )

            # interaction_done-only frame
            if frame_label is None:
                if (
                    person_labels
                    and set(person_labels)
                    == {"interaction_done"}
                ):
                    excluded_done_only += 1

                continue

            # ------------------------------------------------
            # Remap leftover interaction_done boxes
            # ------------------------------------------------

            for person in people:

                if (
                    person["person_label"]
                    == "interaction_done"
                ):
                    person["person_label"] = (
                        "not_interaction_ready"
                    )

            # ------------------------------------------------
            # Keep maximum 5 largest boxes
            # ------------------------------------------------

            people = sorted(
                people,
                key=lambda person: bbox_area(
                    person["bbox"]
                ),
                reverse=True,
            )

            people = people[
                :MAX_BOXES_PER_FRAME
            ]

            frame_key = (
                source_dataset,
                video_name,
                frame_number,
            )

            frames[frame_key] = {
                "source_dataset": source_dataset,
                "video_name": video_name,
                "frame_number": frame_number,
                "frame_label": frame_label,
                "people": people,
            }

    print(
        f"{source_dataset}: "
        f"{len(frames):,} usable frames loaded"
    )

    print(
        f"{source_dataset}: "
        f"{excluded_done_only:,} "
        f"interaction_done-only frames excluded"
    )

    return frames


# ============================================================
# DATAFRAME CREATION
# ============================================================

def frames_to_dataframe(frames):
    """
    Convert selected frames into one row per person/bounding box.
    """

    rows = []

    for frame in frames:

        for person in frame["people"]:

            rows.append(
                {
                    "source_dataset": (
                        frame["source_dataset"]
                    ),
                    "video_name": (
                        frame["video_name"]
                    ),
                    "frame_number": (
                        frame["frame_number"]
                    ),
                    "frame_label": (
                        frame["frame_label"]
                    ),
                    "person_id": (
                        person["person_id"]
                    ),
                    "person_label": (
                        person["person_label"]
                    ),
                    "bbox": json.dumps(
                        person["bbox"]
                    ),
                    "frame_path": (
                        person["frame_path"]
                    ),
                }
            )

    dataframe = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if dataframe.empty:
        return dataframe

    dataframe["_frame_sort"] = (
        dataframe["frame_number"].apply(
            frame_sort_key
        )
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "source_dataset",
                "video_name",
                "_frame_sort",
                "person_id",
            ],
            kind="stable",
        )
        .drop(columns="_frame_sort")
        .reset_index(drop=True)
    )

    return dataframe


# ============================================================
# SELECT INITIAL POSITIVE DATA
# ============================================================

def select_initial_frames(
    ssup_frames,
    avidar_frames,
):
    """
    Initial included data:

    1. All usable SSUP-HRI frames.
    2. All AVIDAR interaction_ready frames.

    Also returns the AVIDAR videos that were used for
    interaction_ready data so that those videos are NOT
    later used as negative-only videos.
    """

    initial_frames = list(
        ssup_frames.values()
    )

    avidar_ready_frames = []

    avidar_ready_videos = set()

    for frame in avidar_frames.values():

        if (
            frame["frame_label"]
            == "interaction_ready"
        ):
            avidar_ready_frames.append(frame)

            avidar_ready_videos.add(
                frame["video_name"]
            )

    initial_frames.extend(
        avidar_ready_frames
    )

    return (
        initial_frames,
        avidar_ready_videos,
    )


# ============================================================
# COUNT UNIQUE FRAME LABELS
# ============================================================

def count_frame_labels(frames):
    """
    Count unique frames per frame label.
    """

    return Counter(
        frame["frame_label"]
        for frame in frames
    )


# ============================================================
# BUILD NEGATIVE VIDEO POOL
# ============================================================

def build_negative_video_pool(
    jrdb_frames,
    avidar_frames,
    avidar_ready_videos,
):
    """
    Build:

        video -> eligible not_interaction_ready frames

    JRDB:
        all videos may contribute negatives.

    AVIDAR:
        only videos NOT already used for interaction_ready
        frames are allowed to contribute negatives.
    """

    pool = defaultdict(list)

    # --------------------------------------------------------
    # JRDB
    # --------------------------------------------------------

    for frame in jrdb_frames.values():

        if (
            frame["frame_label"]
            != "not_interaction_ready"
        ):
            continue

        video_key = (
            "JRDB",
            frame["video_name"],
        )

        pool[video_key].append(frame)

    # --------------------------------------------------------
    # Unused AVIDAR videos
    # --------------------------------------------------------

    for frame in avidar_frames.values():

        if (
            frame["video_name"]
            in avidar_ready_videos
        ):
            continue

        if (
            frame["frame_label"]
            != "not_interaction_ready"
        ):
            continue

        video_key = (
            "AVIDAR",
            frame["video_name"],
        )

        pool[video_key].append(frame)

    # Sort frames chronologically in every video.
    for video_key in pool:

        pool[video_key] = sorted(
            pool[video_key],
            key=lambda frame: frame_sort_key(
                frame["frame_number"]
            ),
        )

    return dict(pool)


# ============================================================
# EQUAL QUOTA ALLOCATION
# ============================================================

def allocate_equal_quotas(
    total_required,
    video_keys,
    rng,
):
    """
    Divide N as equally as possible across videos.

    Example:

        N = 10
        videos = 3

    quotas become:

        4, 3, 3

    The videos receiving the remainder are randomized.
    """

    if total_required <= 0:
        return {
            video_key: 0
            for video_key in video_keys
        }

    num_videos = len(video_keys)

    if num_videos == 0:
        raise ValueError(
            "No eligible JRDB/AVIDAR negative videos "
            "were found."
        )

    base_quota = (
        total_required // num_videos
    )

    remainder = (
        total_required % num_videos
    )

    shuffled_keys = list(video_keys)

    rng.shuffle(shuffled_keys)

    quotas = {
        video_key: base_quota
        for video_key in video_keys
    }

    for video_key in shuffled_keys[:remainder]:
        quotas[video_key] += 1

    return quotas


# ============================================================
# FRAME-BASED NEGATIVE SAMPLING
# ============================================================

def sample_frame_based_negatives(
    negative_pool,
    quotas,
    rng,
):
    """
    Randomly select D independent frames from each video.

    If a video has fewer than D eligible frames,
    all eligible frames are selected.
    """

    selected = []

    for video_key, frames in negative_pool.items():

        target = quotas.get(
            video_key,
            0,
        )

        if target <= 0:
            continue

        if len(frames) <= target:

            chosen = list(frames)

        else:

            chosen = rng.sample(
                frames,
                target,
            )

        selected.extend(chosen)

    return selected


# ============================================================
# TEMPORAL NEGATIVE SAMPLING
# ============================================================

def is_numeric_frame_number(frame):
    try:
        int(frame["frame_number"])
        return True
    except (TypeError, ValueError):
        return False


def find_consecutive_runs(frames):
    """
    Split frames into temporally consecutive runs.

    Example:

        1, 2, 3, 8, 9

    becomes:

        [1,2,3]
        [8,9]
    """

    if not frames:
        return []

    # If frame IDs are not numeric, just treat all sorted
    # frames as one sequence.
    if not all(
        is_numeric_frame_number(frame)
        for frame in frames
    ):
        return [frames]

    sorted_frames = sorted(
        frames,
        key=lambda frame: int(
            frame["frame_number"]
        ),
    )

    runs = []

    current_run = [
        sorted_frames[0]
    ]

    for frame in sorted_frames[1:]:

        previous_number = int(
            current_run[-1]["frame_number"]
        )

        current_number = int(
            frame["frame_number"]
        )

        if (
            current_number
            == previous_number + 1
        ):
            current_run.append(frame)

        else:
            runs.append(current_run)

            current_run = [frame]

    runs.append(current_run)

    return runs


def sample_temporal_sequence(
    frames,
    target,
    rng,
):
    """
    Select one continuous sequence of target frames.

    If the video contains fewer than target eligible frames,
    all eligible frames are returned.

    If enough eligible frames exist but they are split into
    non-consecutive groups, the function selects a valid
    consecutive run whenever one is available.

    If no run reaches target length, the longest available
    run is used.
    """

    if target <= 0:
        return []

    if len(frames) <= target:
        return list(frames)

    runs = find_consecutive_runs(frames)

    valid_runs = [
        run
        for run in runs
        if len(run) >= target
    ]

    if valid_runs:

        chosen_run = rng.choice(
            valid_runs
        )

        max_start = (
            len(chosen_run) - target
        )

        start = rng.randint(
            0,
            max_start,
        )

        return chosen_run[
            start:start + target
        ]

    # No run contains D consecutive negative frames.
    longest_length = max(
        len(run)
        for run in runs
    )

    longest_runs = [
        run
        for run in runs
        if len(run) == longest_length
    ]

    chosen_run = rng.choice(
        longest_runs
    )

    return chosen_run


def sample_temporal_negatives(
    negative_pool,
    quotas,
    rng,
):
    """
    Sample one continuous negative-frame sequence
    from every eligible video.
    """

    selected = []

    for video_key, frames in negative_pool.items():

        target = quotas.get(
            video_key,
            0,
        )

        if target <= 0:
            continue

        chosen = sample_temporal_sequence(
            frames=frames,
            target=target,
            rng=rng,
        )

        selected.extend(chosen)

    return selected


# ============================================================
# STATISTICS
# ============================================================

def create_statistics(
    dataframe,
    model_type,
):
    """
    Create a long-format statistics table containing:

    - frame-level statistics
    - bounding-box-level statistics
    - video-level statistics
    """

    rows = []

    if dataframe.empty:
        return pd.DataFrame()

    frame_key_columns = [
        "source_dataset",
        "video_name",
        "frame_number",
    ]

    # One row per unique frame.
    frame_df = (
        dataframe[
            frame_key_columns
            + ["frame_label"]
        ]
        .drop_duplicates(
            subset=frame_key_columns
        )
    )

    # ========================================================
    # FRAME LEVEL
    # ========================================================

    rows.append(
        {
            "model_type": model_type,
            "statistics_level": "frame",
            "scope": "dataset_total",
            "source_dataset": "",
            "video_name": "",
            "label": "all",
            "count": len(frame_df),
        }
    )

    # Frames per label
    frame_label_counts = (
        frame_df["frame_label"]
        .value_counts()
    )

    for label in LABEL_PRIORITY:

        rows.append(
            {
                "model_type": model_type,
                "statistics_level": "frame",
                "scope": "dataset_label",
                "source_dataset": "",
                "video_name": "",
                "label": label,
                "count": int(
                    frame_label_counts.get(
                        label,
                        0,
                    )
                ),
            }
        )

    # Frames per label per dataset
    grouped = (
        frame_df
        .groupby(
            [
                "source_dataset",
                "frame_label",
            ]
        )
        .size()
    )

    for source_dataset in sorted(
        frame_df["source_dataset"].unique()
    ):

        for label in LABEL_PRIORITY:

            count = int(
                grouped.get(
                    (
                        source_dataset,
                        label,
                    ),
                    0,
                )
            )

            rows.append(
                {
                    "model_type": model_type,
                    "statistics_level": "frame",
                    "scope": "dataset_source_label",
                    "source_dataset": source_dataset,
                    "video_name": "",
                    "label": label,
                    "count": count,
                }
            )

    # Frames per label per video
    video_frame_counts = (
        frame_df
        .groupby(
            [
                "source_dataset",
                "video_name",
                "frame_label",
            ]
        )
        .size()
    )

    videos = (
        frame_df[
            [
                "source_dataset",
                "video_name",
            ]
        ]
        .drop_duplicates()
    )

    for _, video in videos.iterrows():

        source_dataset = (
            video["source_dataset"]
        )

        video_name = (
            video["video_name"]
        )

        for label in LABEL_PRIORITY:

            count = int(
                video_frame_counts.get(
                    (
                        source_dataset,
                        video_name,
                        label,
                    ),
                    0,
                )
            )

            rows.append(
                {
                    "model_type": model_type,
                    "statistics_level": "frame",
                    "scope": "video_label",
                    "source_dataset": source_dataset,
                    "video_name": video_name,
                    "label": label,
                    "count": count,
                }
            )

    # ========================================================
    # BOUNDING BOX LEVEL
    # ========================================================

    rows.append(
        {
            "model_type": model_type,
            "statistics_level": "bounding_box",
            "scope": "dataset_total",
            "source_dataset": "",
            "video_name": "",
            "label": "all",
            "count": len(dataframe),
        }
    )

    person_label_counts = (
        dataframe["person_label"]
        .value_counts()
    )

    for label in LABEL_PRIORITY:

        rows.append(
            {
                "model_type": model_type,
                "statistics_level": "bounding_box",
                "scope": "dataset_label",
                "source_dataset": "",
                "video_name": "",
                "label": label,
                "count": int(
                    person_label_counts.get(
                        label,
                        0,
                    )
                ),
            }
        )

    # Bounding boxes by dataset + label
    box_dataset_counts = (
        dataframe
        .groupby(
            [
                "source_dataset",
                "person_label",
            ]
        )
        .size()
    )

    for source_dataset in sorted(
        dataframe["source_dataset"].unique()
    ):

        for label in LABEL_PRIORITY:

            count = int(
                box_dataset_counts.get(
                    (
                        source_dataset,
                        label,
                    ),
                    0,
                )
            )

            rows.append(
                {
                    "model_type": model_type,
                    "statistics_level": "bounding_box",
                    "scope": "dataset_source_label",
                    "source_dataset": source_dataset,
                    "video_name": "",
                    "label": label,
                    "count": count,
                }
            )

    # Bounding boxes by video + label
    box_video_counts = (
        dataframe
        .groupby(
            [
                "source_dataset",
                "video_name",
                "person_label",
            ]
        )
        .size()
    )

    for _, video in videos.iterrows():

        source_dataset = (
            video["source_dataset"]
        )

        video_name = (
            video["video_name"]
        )

        for label in LABEL_PRIORITY:

            count = int(
                box_video_counts.get(
                    (
                        source_dataset,
                        video_name,
                        label,
                    ),
                    0,
                )
            )

            rows.append(
                {
                    "model_type": model_type,
                    "statistics_level": "bounding_box",
                    "scope": "video_label",
                    "source_dataset": source_dataset,
                    "video_name": video_name,
                    "label": label,
                    "count": count,
                }
            )

    # ========================================================
    # VIDEO LEVEL
    # ========================================================

    total_videos = len(videos)

    rows.append(
        {
            "model_type": model_type,
            "statistics_level": "video",
            "scope": "dataset_total",
            "source_dataset": "",
            "video_name": "",
            "label": "all",
            "count": total_videos,
        }
    )

    videos_per_dataset = (
        videos["source_dataset"]
        .value_counts()
    )

    for source_dataset, count in (
        videos_per_dataset.items()
    ):

        rows.append(
            {
                "model_type": model_type,
                "statistics_level": "video",
                "scope": "source_dataset_total",
                "source_dataset": source_dataset,
                "video_name": "",
                "label": "all",
                "count": int(count),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_dataset_summary(
    name,
    frames,
):
    """
    Print frame-level summary.
    """

    counts = count_frame_labels(frames)

    print(
        f"\n========== {name} =========="
    )

    print(
        f"Total unique frames: "
        f"{len(frames):,}"
    )

    for label in LABEL_PRIORITY:

        print(
            f"{label}: "
            f"{counts.get(label, 0):,}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Separate RNG instances so each output is reproducible.
    frame_rng = random.Random(
        RANDOM_SEED
    )

    temporal_rng = random.Random(
        RANDOM_SEED
    )

    print(
        "\n========================================"
    )
    print("LOADING DATASETS")
    print(
        "========================================"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    ssup_frames = load_dataset_jsons(
        json_directory=SSUP_JSON_DIR,
        source_dataset="SSUP-HRI",
    )

    avidar_frames = load_dataset_jsons(
        json_directory=AVIDAR_JSON_DIR,
        source_dataset="AVIDAR",
    )

    jrdb_frames = load_dataset_jsons(
        json_directory=JRDB_JSON_DIR,
        source_dataset="JRDB",
    )

    # ========================================================
    # INITIAL INCLUDED DATA
    # ========================================================

    (
        initial_frames,
        avidar_ready_videos,
    ) = select_initial_frames(
        ssup_frames=ssup_frames,
        avidar_frames=avidar_frames,
    )

    initial_counts = count_frame_labels(
        initial_frames
    )

    num_ready = initial_counts.get(
        "interaction_ready",
        0,
    )

    num_not_ready = initial_counts.get(
        "not_interaction_ready",
        0,
    )

    num_ongoing = initial_counts.get(
        "interaction_ongoing",
        0,
    )

    print(
        "\n========================================"
    )
    print("INITIAL INCLUDED DATA")
    print(
        "========================================"
    )

    print(
        f"interaction_ready: "
        f"{num_ready:,}"
    )

    print(
        f"not_interaction_ready: "
        f"{num_not_ready:,}"
    )

    print(
        f"interaction_ongoing: "
        f"{num_ongoing:,}"
    )

    # ========================================================
    # CALCULATE N
    # ========================================================

    N = max(
        0,
        num_ready - num_not_ready,
    )

    print(
        f"\nAdditional not_interaction_ready "
        f"frames required (N): {N:,}"
    )

    # ========================================================
    # BUILD NEGATIVE POOL
    # ========================================================

    negative_pool = build_negative_video_pool(
        jrdb_frames=jrdb_frames,
        avidar_frames=avidar_frames,
        avidar_ready_videos=avidar_ready_videos,
    )

    video_keys = sorted(
        negative_pool.keys()
    )

    print(
        f"Eligible negative videos: "
        f"{len(video_keys):,}"
    )

    if video_keys:

        D = (
            N / len(video_keys)
            if N > 0
            else 0
        )

        print(
            f"Approximate D per video: "
            f"{D:.2f}"
        )

    # ========================================================
    # ALLOCATE QUOTAS
    # ========================================================

    quotas = allocate_equal_quotas(
        total_required=N,
        video_keys=video_keys,
        rng=frame_rng,
    )

    # ========================================================
    # FRAME-BASED SAMPLING
    # ========================================================

    frame_negative_samples = (
        sample_frame_based_negatives(
            negative_pool=negative_pool,
            quotas=quotas,
            rng=frame_rng,
        )
    )

    frame_based_frames = (
        list(initial_frames)
        + frame_negative_samples
    )

    # ========================================================
    # TEMPORAL SAMPLING
    # ========================================================

    # Use same quota values so the target contribution of
    # each video is identical between model types.

    temporal_negative_samples = (
        sample_temporal_negatives(
            negative_pool=negative_pool,
            quotas=quotas,
            rng=temporal_rng,
        )
    )

    temporal_frames = (
        list(initial_frames)
        + temporal_negative_samples
    )

    # ========================================================
    # CREATE OUTPUT DATAFRAMES
    # ========================================================

    frame_based_df = (
        frames_to_dataframe(
            frame_based_frames
        )
    )

    temporal_df = (
        frames_to_dataframe(
            temporal_frames
        )
    )

    # ========================================================
    # CREATE STATISTICS
    # ========================================================

    frame_stats_df = create_statistics(
        dataframe=frame_based_df,
        model_type="frame_based",
    )

    temporal_stats_df = create_statistics(
        dataframe=temporal_df,
        model_type="temporal",
    )

    # ========================================================
    # SAVE
    # ========================================================

    frame_based_df.to_csv(
        FRAME_DATASET_OUTPUT,
        index=False,
    )

    frame_stats_df.to_csv(
        FRAME_STATS_OUTPUT,
        index=False,
    )

    temporal_df.to_csv(
        TEMPORAL_DATASET_OUTPUT,
        index=False,
    )

    temporal_stats_df.to_csv(
        TEMPORAL_STATS_OUTPUT,
        index=False,
    )

    # ========================================================
    # PRINT FINAL STATISTICS
    # ========================================================

    print_dataset_summary(
        "FRAME-BASED DATASET",
        frame_based_frames,
    )

    print_dataset_summary(
        "TEMPORAL DATASET",
        temporal_frames,
    )

    print(
        "\n========================================"
    )
    print("NEGATIVE SAMPLING")
    print(
        "========================================"
    )

    print(
        f"Requested additional negatives: "
        f"{N:,}"
    )

    print(
        f"Frame-based negatives selected: "
        f"{len(frame_negative_samples):,}"
    )

    print(
        f"Temporal negatives selected: "
        f"{len(temporal_negative_samples):,}"
    )

    if (
        len(frame_negative_samples)
        < N
    ):
        print(
            "\nWARNING:"
            "\nThe frame-based dataset contains fewer "
            "additional negatives than requested."
            "\nThis occurs when some videos contain fewer "
            "than their allocated D frames."
        )

    if (
        len(temporal_negative_samples)
        < N
    ):
        print(
            "\nWARNING:"
            "\nThe temporal dataset contains fewer "
            "additional negatives than requested."
            "\nThis can occur when videos contain fewer "
            "than D frames or do not contain a sufficiently "
            "long consecutive negative sequence."
        )

    # ========================================================
    # MISSING FRAME PATHS
    # ========================================================

    frame_missing_paths = (
        frame_based_df["frame_path"]
        .eq("")
        .sum()
    )

    temporal_missing_paths = (
        temporal_df["frame_path"]
        .eq("")
        .sum()
    )

    print(
        "\n========================================"
    )
    print("FRAME PATH CHECK")
    print(
        "========================================"
    )

    print(
        f"Frame-based rows with missing frame_path: "
        f"{frame_missing_paths:,}"
    )

    print(
        f"Temporal rows with missing frame_path: "
        f"{temporal_missing_paths:,}"
    )

    # ========================================================
    # OUTPUT LOCATIONS
    # ========================================================

    print(
        "\n========================================"
    )
    print("SAVED OUTPUTS")
    print(
        "========================================"
    )

    print(
        f"\nFrame-based dataset:\n"
        f"{FRAME_DATASET_OUTPUT}"
    )

    print(
        f"\nFrame-based statistics:\n"
        f"{FRAME_STATS_OUTPUT}"
    )

    print(
        f"\nTemporal dataset:\n"
        f"{TEMPORAL_DATASET_OUTPUT}"
    )

    print(
        f"\nTemporal statistics:\n"
        f"{TEMPORAL_STATS_OUTPUT}"
    )

    print(
        "\n========================================"
    )


if __name__ == "__main__":
    main()