"""
Temporal / sequential data balancing.

Frame count rules:
- We don't use frames that include only 'interaction_done' bounding boxes, since they resemble
  'not_interaction_ready' too much.
- If a frame only contains bounding boxes labelled as 'not_interaction_ready', it will be counted
  as a 'not_interaction_ready' sample.
- If a frame contains at least one bounding box labelled as 'interaction_ready', it will be counted
  as an 'interaction_ready' sample, regardless of the presence of any 'not_interaction_ready' or
  'interaction_ongoing' boxes.
- If a frame contains at least one bounding box labelled as 'interaction_ongoing' and no bounding
  boxes labelled as 'interaction_ready', it will be counted as an 'interaction_ongoing' sample,
  regardless of the presence of any 'not_interaction_ready' boxes.

Data balancing rules:
- All SSUP-HRI frames are included in the dataset since they provide most of the
  interaction_ready and interaction_ongoing labels.
- All AVIDAR 'interaction_ready' frames are included in the dataset since they provide a
  different type of interaction_ready frames.
- Remove frames that include only 'interaction_done' bounding boxes.
- Rename leftover 'interaction_done' bounding boxes to 'not_interaction_ready'.
- The currently included frames have more interaction_ready than not_interaction_ready samples,
  so include not_interaction_ready frames from JRDB videos and currently unused AVIDAR videos.
- The number of not_interaction_ready frames to include corresponds to the difference between the
  number of interaction_ready frames and the number of not_interaction_ready frames in the
  currently included frames. Designate this number as N.
- Divide N as equally as possible across the total number of eligible leftover JRDB and AVIDAR videos.
- For every eligible negative video, randomly select one continuous sequence of its allocated
  number of frames.
- If the video has fewer eligible frames than its allocated quota, include all of them.
- If no consecutive run is long enough to satisfy the quota, use the longest consecutive run.

Person selection for temporal models:
- Person selection is done at VIDEO level, not independently per frame.
- For each complete video, calculate the average bounding-box area of every person across all
  usable frames in which that person appears.
- Rank people by decreasing average bounding-box area.
- Keep only the 5 people with the largest average bounding-box areas.
- Reassign their IDs consistently over the whole video:
      person_id 1 = largest average bbox area
      person_id 2 = second-largest average bbox area
      ...
      person_id 5 = fifth-largest average bbox area
- People outside the top 5 are discarded from every frame of that video.
- If a video contains fewer than 5 people, keep all available people and assign IDs from 1 upward
  in decreasing average-area order.

Output:
- balanced_temporal_dataset.csv
- balanced_temporal_dataset_statistics.csv
"""

from pathlib import Path
import ast
import json
import random
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SSUP_JSON_DIR = Path(
    r"/home/marilyn/Downloads/data/JSON_hierarchical/SSUP-HRI/AstorPlace_final"
)

AVIDAR_JSON_DIR = Path(
    r"/home/marilyn/Downloads/data/JSON_hierarchical/AVIDAR"
)

JRDB_JSON_DIR = Path(
    r"/home/marilyn/Downloads/data/JSON_hierarchical/JRDB"
)


FRAME_ROOTS = {
    "SSUP-HRI": Path(
        r"/home/marilyn/Downloads/data/frames/SSUP-HRI"
    ),
    "AVIDAR": Path(
        r"/home/marilyn/Downloads/data/frames/AVIDAR"
    ),
    "JRDB": Path(
        r"/home/marilyn/Downloads/data/frames/JRDB"
    ),
}


OUTPUT_DIR = Path(
    r"/home/marilyn/Downloads/data"
)

TEMPORAL_DATASET_OUTPUT = (
    OUTPUT_DIR / "balanced_temporal_dataset.csv"
)

TEMPORAL_STATS_OUTPUT = (
    OUTPUT_DIR / "balanced_temporal_dataset_statistics.csv"
)


RANDOM_SEED = 42
MAX_PEOPLE_PER_VIDEO = 5


# ============================================================
# CONSTANTS
# ============================================================

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
    Sorting key that handles both numeric and non-numeric frame IDs.
    """

    try:
        return 0, int(frame_number)
    except (TypeError, ValueError):
        return 1, str(frame_number)


def person_sort_key(person_id):
    """
    Deterministic secondary sorting key for person IDs.

    Used only to break ties when two people have exactly the same
    average bounding-box area.
    """

    try:
        return 0, int(person_id)
    except (TypeError, ValueError):
        return 1, str(person_id)


def bbox_area(bbox):
    """
    Calculate bounding-box area.

    Expected bbox format:
        [x, y, width, height]

    Returns 0 if the bbox is malformed.
    """

    if bbox is None:
        return 0.0

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

        return (
            max(0.0, width)
            * max(0.0, height)
        )

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
    3. Use image if present.
    4. Try common filenames under FRAME_ROOTS.
    """

    for key in [
        "frame_path",
        "image_path",
        "image",
    ]:
        value = annotation.get(key)

        if isinstance(value, str) and value.strip():
            return value

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

    return ""


# ============================================================
# FRAME LABEL AGGREGATION
# ============================================================

def determine_frame_label(person_labels):
    """
    Determine one label for an entire frame.

    Priority:
        interaction_ready
        interaction_ongoing
        not_interaction_ready

    interaction_done-only frames are excluded.
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

    IMPORTANT:
    No top-5 filtering is performed here.

    All people must remain available until average person bbox areas
    have been calculated over the whole video.
    """

    json_paths = sorted(
        json_directory.glob("*.json")
    )

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

                people_by_frame[
                    frame_number
                ].append(
                    {
                        "person_id": str(person_id),
                        "person_label": label,
                        "bbox": bbox,
                        "frame_path": frame_path,
                    }
                )

        for frame_number, people in people_by_frame.items():

            person_labels = [
                person["person_label"]
                for person in people
            ]

            frame_label = determine_frame_label(
                person_labels
            )

            if frame_label is None:

                if (
                    person_labels
                    and set(person_labels)
                    == {"interaction_done"}
                ):
                    excluded_done_only += 1

                continue

            # Remap leftover interaction_done boxes.
            for person in people:

                if (
                    person["person_label"]
                    == "interaction_done"
                ):
                    person["person_label"] = (
                        "not_interaction_ready"
                    )

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
# VIDEO-LEVEL PERSON RANKING
# ============================================================

def build_temporal_person_rankings(
    ssup_frames,
    avidar_frames,
    jrdb_frames,
):
    """
    Calculate the average bbox area of every person over the whole
    usable video.

    Then select at most five people per video and assign:

        new ID 1 -> largest mean bbox area
        new ID 2 -> second largest
        ...
        new ID 5 -> fifth largest

    Returns
    -------
    rankings : dict
        {
            (source_dataset, video_name): {
                original_person_id: new_person_id
            }
        }

    mean_areas_by_video : dict
        Mean bbox areas, retained mainly for logging/verification.
    """

    areas_by_video_person = defaultdict(
        lambda: defaultdict(list)
    )

    all_frames = (
        list(ssup_frames.values())
        + list(avidar_frames.values())
        + list(jrdb_frames.values())
    )

    for frame in all_frames:

        video_key = (
            frame["source_dataset"],
            frame["video_name"],
        )

        for person in frame["people"]:

            original_person_id = str(
                person["person_id"]
            )

            areas_by_video_person[
                video_key
            ][original_person_id].append(
                bbox_area(
                    person["bbox"]
                )
            )

    rankings = {}
    mean_areas_by_video = {}

    for (
        video_key,
        person_areas,
    ) in areas_by_video_person.items():

        mean_areas = {}

        for (
            person_id,
            areas,
        ) in person_areas.items():

            mean_areas[person_id] = (
                sum(areas) / len(areas)
                if areas
                else 0.0
            )

        mean_areas_by_video[
            video_key
        ] = mean_areas

        ranked_people = sorted(
            mean_areas.items(),
            key=lambda item: (
                -item[1],
                person_sort_key(item[0]),
            ),
        )

        top_people = ranked_people[
            :MAX_PEOPLE_PER_VIDEO
        ]

        rankings[video_key] = {
            original_person_id: str(rank)
            for rank, (
                original_person_id,
                _
            ) in enumerate(
                top_people,
                start=1,
            )
        }

    return (
        rankings,
        mean_areas_by_video,
    )


def apply_temporal_person_selection(
    frames,
    person_rankings,
):
    """
    Apply the precomputed video-level top-5 identity mapping.

    Only selected identities are retained.

    The original frame_label is preserved so that the labels used
    during balancing are not retroactively changed after person
    filtering.
    """

    processed_frames = []

    for frame in frames:

        video_key = (
            frame["source_dataset"],
            frame["video_name"],
        )

        ranking = person_rankings.get(
            video_key,
            {},
        )

        selected_people = []

        for person in frame["people"]:

            original_person_id = str(
                person["person_id"]
            )

            if original_person_id not in ranking:
                continue

            new_person = dict(person)

            new_person["person_id"] = (
                ranking[
                    original_person_id
                ]
            )

            selected_people.append(
                new_person
            )

        selected_people = sorted(
            selected_people,
            key=lambda person: int(
                person["person_id"]
            ),
        )

        # Since the output CSV has one row per person, a frame with
        # none of the selected top-5 people cannot be represented.
        if not selected_people:
            continue

        processed_frames.append(
            {
                **frame,
                "people": selected_people,
            }
        )

    return processed_frames


def print_person_rankings(
    rankings,
    mean_areas_by_video,
):
    """
    Print the selected identities and their new person IDs.
    """

    print(
        "\n========================================"
    )
    print("VIDEO-LEVEL PERSON RANKINGS")
    print(
        "========================================"
    )

    for video_key in sorted(
        rankings.keys()
    ):

        source_dataset, video_name = (
            video_key
        )

        print(
            f"\n{source_dataset} / {video_name}"
        )

        ranked_people = sorted(
            rankings[video_key].items(),
            key=lambda item: int(
                item[1]
            ),
        )

        for (
            original_person_id,
            new_person_id,
        ) in ranked_people:

            mean_area = (
                mean_areas_by_video[
                    video_key
                ][original_person_id]
            )

            print(
                f"  new person_id {new_person_id} "
                f"<- original person_id {original_person_id} "
                f"| average bbox area = {mean_area:.2f}"
            )


# ============================================================
# INITIAL INCLUDED DATA
# ============================================================

def select_initial_frames(
    ssup_frames,
    avidar_frames,
):
    """
    Initial included data:

    1. All usable SSUP-HRI frames.
    2. All AVIDAR interaction_ready frames.

    Returns the AVIDAR videos used for interaction_ready data
    so that they are not reused as negative-only videos.
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
            avidar_ready_frames.append(
                frame
            )

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
# COUNT FRAME LABELS
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
# NEGATIVE VIDEO POOL
# ============================================================

def build_negative_video_pool(
    jrdb_frames,
    avidar_frames,
    avidar_ready_videos,
):
    """
    Build:
        (source_dataset, video_name) -> eligible not_interaction_ready frames

    JRDB:
        all videos may contribute negative frames.

    AVIDAR:
        only videos not already used for interaction_ready samples
        may contribute negative frames.
    """

    pool = defaultdict(list)

    # JRDB negatives
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

        pool[video_key].append(
            frame
        )

    # Unused AVIDAR negatives
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

        pool[video_key].append(
            frame
        )

    # Sort every video's candidate frames chronologically.
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
    Divide N as equally as possible across eligible videos.

    Example:
        N = 10
        videos = 3

    quotas:
        4, 3, 3

    Which video receives the extra frame is randomized but
    reproducible through RANDOM_SEED.
    """

    if total_required <= 0:
        return {
            video_key: 0
            for video_key in video_keys
        }

    num_videos = len(
        video_keys
    )

    if num_videos == 0:
        raise ValueError(
            "No eligible JRDB/AVIDAR negative videos were found."
        )

    base_quota = (
        total_required
        // num_videos
    )

    remainder = (
        total_required
        % num_videos
    )

    shuffled_keys = list(
        video_keys
    )

    rng.shuffle(
        shuffled_keys
    )

    quotas = {
        video_key: base_quota
        for video_key in video_keys
    }

    for video_key in (
        shuffled_keys[:remainder]
    ):
        quotas[video_key] += 1

    return quotas


# ============================================================
# TEMPORAL NEGATIVE SAMPLING
# ============================================================

def is_numeric_frame_number(frame):
    """
    Check whether a frame number can be interpreted as an integer.
    """

    try:
        int(
            frame["frame_number"]
        )
        return True

    except (TypeError, ValueError):
        return False


def find_consecutive_runs(frames):
    """
    Split frames into temporally consecutive runs.

    Example:
        1, 2, 3, 8, 9

    becomes:
        [1, 2, 3]
        [8, 9]
    """

    if not frames:
        return []

    if not all(
        is_numeric_frame_number(frame)
        for frame in frames
    ):
        # If frame IDs are non-numeric, preserve the sorted
        # sequence as one run because exact adjacency cannot
        # be inferred from the IDs.
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
            current_run[-1][
                "frame_number"
            ]
        )

        current_number = int(
            frame[
                "frame_number"
            ]
        )

        if (
            current_number
            == previous_number + 1
        ):
            current_run.append(
                frame
            )

        else:
            runs.append(
                current_run
            )

            current_run = [
                frame
            ]

    runs.append(
        current_run
    )

    return runs


def sample_temporal_sequence(
    frames,
    target,
    rng,
):
    """
    Select one continuous sequence of target negative frames.

    If the video contains fewer than target eligible frames,
    return all eligible frames.

    If multiple runs can satisfy the requested length,
    randomly choose one run and then randomly choose the
    starting point within that run.

    If no consecutive run reaches target length, return one
    of the longest available runs.
    """

    if target <= 0:
        return []

    if len(frames) <= target:
        return list(
            frames
        )

    runs = find_consecutive_runs(
        frames
    )

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
            len(chosen_run)
            - target
        )

        start = rng.randint(
            0,
            max_start,
        )

        return chosen_run[
            start:start + target
        ]

    longest_length = max(
        len(run)
        for run in runs
    )

    longest_runs = [
        run
        for run in runs
        if len(run)
        == longest_length
    ]

    return rng.choice(
        longest_runs
    )


def sample_temporal_negatives(
    negative_pool,
    quotas,
    rng,
):
    """
    Sample one continuous negative sequence from every
    eligible video.
    """

    selected = []

    for (
        video_key,
        frames,
    ) in negative_pool.items():

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

        selected.extend(
            chosen
        )

    return selected


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
                        frame[
                            "source_dataset"
                        ]
                    ),
                    "video_name": (
                        frame[
                            "video_name"
                        ]
                    ),
                    "frame_number": (
                        frame[
                            "frame_number"
                        ]
                    ),
                    "frame_label": (
                        frame[
                            "frame_label"
                        ]
                    ),
                    "person_id": (
                        person[
                            "person_id"
                        ]
                    ),
                    "person_label": (
                        person[
                            "person_label"
                        ]
                    ),
                    "bbox": json.dumps(
                        person[
                            "bbox"
                        ]
                    ),
                    "frame_path": (
                        person[
                            "frame_path"
                        ]
                    ),
                }
            )

    dataframe = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    if dataframe.empty:
        return dataframe

    dataframe[
        "_frame_sort"
    ] = (
        dataframe[
            "frame_number"
        ].apply(
            frame_sort_key
        )
    )

    dataframe[
        "_person_sort"
    ] = pd.to_numeric(
        dataframe[
            "person_id"
        ],
        errors="coerce",
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "source_dataset",
                "video_name",
                "_frame_sort",
                "_person_sort",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_frame_sort",
                "_person_sort",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


# ============================================================
# STATISTICS
# ============================================================

def create_statistics(
    dataframe,
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
            "statistics_level": "frame",
            "scope": "dataset_total",
            "source_dataset": "",
            "video_name": "",
            "label": "all",
            "count": len(
                frame_df
            ),
        }
    )

    frame_label_counts = (
        frame_df[
            "frame_label"
        ]
        .value_counts()
    )

    for label in LABEL_PRIORITY:

        rows.append(
            {
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
        frame_df[
            "source_dataset"
        ].unique()
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
                    "statistics_level": "frame",
                    "scope": "dataset_source_label",
                    "source_dataset": source_dataset,
                    "video_name": "",
                    "label": label,
                    "count": count,
                }
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

    for _, video in videos.iterrows():

        source_dataset = (
            video[
                "source_dataset"
            ]
        )

        video_name = (
            video[
                "video_name"
            ]
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
            "statistics_level": "bounding_box",
            "scope": "dataset_total",
            "source_dataset": "",
            "video_name": "",
            "label": "all",
            "count": len(
                dataframe
            ),
        }
    )

    person_label_counts = (
        dataframe[
            "person_label"
        ]
        .value_counts()
    )

    for label in LABEL_PRIORITY:

        rows.append(
            {
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
        dataframe[
            "source_dataset"
        ].unique()
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
                    "statistics_level": "bounding_box",
                    "scope": "dataset_source_label",
                    "source_dataset": source_dataset,
                    "video_name": "",
                    "label": label,
                    "count": count,
                }
            )

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
            video[
                "source_dataset"
            ]
        )

        video_name = (
            video[
                "video_name"
            ]
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

    total_videos = len(
        videos
    )

    rows.append(
        {
            "statistics_level": "video",
            "scope": "dataset_total",
            "source_dataset": "",
            "video_name": "",
            "label": "all",
            "count": total_videos,
        }
    )

    videos_per_dataset = (
        videos[
            "source_dataset"
        ]
        .value_counts()
    )

    for (
        source_dataset,
        count,
    ) in videos_per_dataset.items():

        rows.append(
            {
                "statistics_level": "video",
                "scope": "source_dataset_total",
                "source_dataset": source_dataset,
                "video_name": "",
                "label": "all",
                "count": int(
                    count
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_dataset_summary(
    frames,
):
    """
    Print frame-level summary.
    """

    counts = count_frame_labels(
        frames
    )

    print(
        "\n========== TEMPORAL DATASET =========="
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

    rng = random.Random(
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
    # VIDEO-LEVEL PERSON RANKINGS
    # ========================================================

    (
        person_rankings,
        mean_areas_by_video,
    ) = build_temporal_person_rankings(
        ssup_frames=ssup_frames,
        avidar_frames=avidar_frames,
        jrdb_frames=jrdb_frames,
    )

    print_person_rankings(
        rankings=person_rankings,
        mean_areas_by_video=mean_areas_by_video,
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
    # BUILD NEGATIVE VIDEO POOL
    # ========================================================

    negative_pool = (
        build_negative_video_pool(
            jrdb_frames=jrdb_frames,
            avidar_frames=avidar_frames,
            avidar_ready_videos=(
                avidar_ready_videos
            ),
        )
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
        rng=rng,
    )

    # ========================================================
    # TEMPORAL NEGATIVE SAMPLING
    # ========================================================

    temporal_negative_samples = (
        sample_temporal_negatives(
            negative_pool=negative_pool,
            quotas=quotas,
            rng=rng,
        )
    )

    # ========================================================
    # COMBINE DATA
    # ========================================================

    temporal_frames = (
        list(initial_frames)
        + temporal_negative_samples
    )

    # ========================================================
    # APPLY VIDEO-LEVEL TOP-5 PERSON SELECTION
    # ========================================================

    temporal_frames = (
        apply_temporal_person_selection(
            frames=temporal_frames,
            person_rankings=person_rankings,
        )
    )

    # ========================================================
    # CREATE OUTPUT DATAFRAME
    # ========================================================

    temporal_df = (
        frames_to_dataframe(
            temporal_frames
        )
    )

    # ========================================================
    # CREATE STATISTICS
    # ========================================================

    temporal_stats_df = (
        create_statistics(
            dataframe=temporal_df
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    temporal_df.to_csv(
        TEMPORAL_DATASET_OUTPUT,
        index=False,
    )

    temporal_stats_df.to_csv(
        TEMPORAL_STATS_OUTPUT,
        index=False,
    )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    print_dataset_summary(
        temporal_frames
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
        f"Temporal negatives selected: "
        f"{len(temporal_negative_samples):,}"
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
            "than their allocated number of eligible frames "
            "or do not contain a sufficiently long "
            "consecutive negative sequence."
        )

    # ========================================================
    # MISSING FRAME PATHS
    # ========================================================

    temporal_missing_paths = (
        temporal_df[
            "frame_path"
        ]
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
        f"Rows with missing frame_path: "
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
