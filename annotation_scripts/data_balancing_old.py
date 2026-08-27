"""
Build the balanced Interaction Readiness dataset.

Expected input columns:
    source_dataset, video_id, person_id, frame_id, image, bbox, label

Selection strategy
------------------
Frames per label:

JRDB:
  not_interaction_ready: 243307
AVIDAR:
  interaction_ready: 2404
  not_interaction_ready: 75537
SSUP-HRI:
  interaction_ready: 21448
  not_interaction_ready: 13200
  interaction_ongoing: 5352
  interaction_done: 12882

Strategy:
- Do not use interaction_done frames, as they are similar to not_interaction_ready, 
    with SSUP specific details that might not generalize.
- Include all of SSUP-HRI, which has all labels in most videos/people.
- Include all of the AVIDAR interaction_ready frames since they constitute another variety of interaction_ready frames (== 4 videos, 6 people).
- For now we have:
    interaction_ready: 23852
    not_interaction_ready: 13200
    interaction_ongoing: 5352
- interaction_ongoing is less important, as robot won't need to detect interaction readiness while interaction is ongoing, 
    so we can keep the interaction_ongoing samples fewer.
- the essential classes are interaction_ready and not_interaction_ready, so we need to balance them.
- use 10,652 not_interaction_ready frames distributed equally between JRDB and AVIDAR to balance the dataset, which means:
    - 5,326 frames from JRDB not_interaction_ready
    - 5,326 frames from AVIDAR not_interaction_ready
- to have as much diversity as possible:
    - we will not select more than one person per video (maximise the included settings)
    - we will pick the people that have the least frames in the datasets, to maximise the number of people included in the dataset
        - but also only keep people that have at least 100 frames, to make sure the sequence is informative enough for the model to learn from it.
- use the above rules to select person IDs from videos until 5326 frames per dataset are selected.

Outputs:
    balanced_dataset.parquet
    selected_people.csv
    dataset_statistics.csv
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# ==============================================================================
# CONFIGURATION — EDIT THESE VALUES
# ==============================================================================

JRDB_PARQUET = Path(
    r"D:\EngagementDetection\Push_to_HF\parquet_final_gated\JRDB.parquet"
)
AVIDAR_PARQUET = Path(
    r"D:\EngagementDetection\Push_to_HF\parquet_final_gated\AVIDAR.parquet"
)
SSUP_PARQUET = Path(
    r"D:\EngagementDetection\Push_to_HF\parquet_final_gated\SSUP-HRI.parquet"
)

OUTPUT_DIR = Path(
    r"D:\EngagementDetection\Push_to_HF\parquet_final_gated\balanced_and_split"
)

TARGET_EXTRA_NIR = 5326
MIN_PERSON_FRAMES = 100
RANDOM_SEED = 42


# ==============================================================================
# CONSTANTS
# ==============================================================================

REQUIRED_COLUMNS = {
    "image",
    "source_dataset",
    "video_id",
    "person_id",
    "frame_id",
    "bbox",
    "label",
}

METADATA_COLUMNS = [
    "source_dataset",
    "video_id",
    "person_id",
    "frame_id",
    "label",
]

LABEL_NOT_READY = "not_interaction_ready"
LABEL_READY = "interaction_ready"
LABEL_ONGOING = "interaction_ongoing"
LABEL_DONE = "interaction_done"

ALLOWED_INPUT_LABELS = {
    LABEL_NOT_READY,
    LABEL_READY,
    LABEL_ONGOING,
    LABEL_DONE,
}

FINAL_LABELS = {
    LABEL_NOT_READY,
    LABEL_READY,
    LABEL_ONGOING,
}


@dataclass(frozen=True)
class Candidate:
    video_id: str
    person_id: str
    num_rows: int


# ==============================================================================
# LOADING
# ==============================================================================

def load_dataset_metadata(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load only lightweight columns used by the balancing logic.

    `source_row_index` is the row's zero-based position in its source Parquet.
    It allows the exact original row to be recovered later.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} Parquet was not found: {path}"
        )

    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema_arrow.names)

    missing = REQUIRED_COLUMNS - available_columns
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {sorted(missing)}"
        )

    df = pd.read_parquet(
        path,
        columns=METADATA_COLUMNS,
    ).copy()

    df["source_row_index"] = range(len(df))
    df["source_dataset"] = dataset_name
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["person_id"] = df["person_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    unexpected_labels = sorted(
        set(df["label"].unique()) - ALLOWED_INPUT_LABELS
    )
    if unexpected_labels:
        raise ValueError(
            f"{dataset_name} contains unsupported labels: "
            f"{unexpected_labels}"
        )

    df["_frame_sort"] = pd.to_numeric(
        df["frame_id"],
        errors="coerce",
    )
    fallback = df.groupby(
        ["video_id", "person_id"],
        sort=False,
    ).cumcount()
    df["_frame_sort"] = df["_frame_sort"].fillna(fallback)

    return df


# ==============================================================================
# NOT-INTERACTION-READY SAMPLING
# ==============================================================================

def choose_one_person_per_video(
    df: pd.DataFrame,
    min_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, list[Candidate]]:
    nir = df.loc[
        df["label"] == LABEL_NOT_READY
    ].copy()

    counts = (
        nir.groupby(
            ["video_id", "person_id"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "num_rows"})
    )

    eligible = counts.loc[
        counts["num_rows"] >= min_rows
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"No eligible people have at least {min_rows} "
            f"{LABEL_NOT_READY} rows."
        )

    def tie_value(row: pd.Series) -> int:
        value = f"{seed}::{row['video_id']}::{row['person_id']}"
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    eligible["_tie"] = eligible.apply(tie_value, axis=1)

    chosen = (
        eligible.sort_values(
            ["video_id", "num_rows", "_tie", "person_id"]
        )
        .drop_duplicates("video_id", keep="first")
        .drop(columns="_tie")
        .reset_index(drop=True)
    )

    candidates = [
        Candidate(
            video_id=row.video_id,
            person_id=row.person_id,
            num_rows=int(row.num_rows),
        )
        for row in chosen.itertuples(index=False)
    ]

    return nir, candidates


def choose_candidates_to_target(
    candidates: list[Candidate],
    target: int,
) -> tuple[list[Candidate], Candidate | None, int]:
    """
    Keep whole people whenever possible, then use one partial final person.
    """
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.num_rows,
            item.video_id,
            item.person_id,
        ),
    )

    if sum(item.num_rows for item in ordered) < target:
        raise ValueError(
            f"Only {sum(item.num_rows for item in ordered)} eligible rows "
            f"are available for target {target}."
        )

    whole: list[Candidate] = []
    skipped: list[Candidate] = []
    total = 0

    for candidate in ordered:
        if total + candidate.num_rows <= target:
            whole.append(candidate)
            total += candidate.num_rows
        else:
            skipped.append(candidate)

    remainder = target - total
    if remainder == 0:
        return whole, None, 0

    partial = next(
        (
            candidate
            for candidate in skipped
            if candidate.num_rows >= remainder
        ),
        None,
    )

    if partial is None:
        raise RuntimeError(
            "No remaining candidate can supply the final remainder."
        )

    return whole, partial, remainder


def centered_segment(
    rows: pd.DataFrame,
    size: int,
) -> pd.DataFrame:
    rows = rows.sort_values(
        ["_frame_sort", "source_row_index"]
    ).copy()

    if size > len(rows):
        raise ValueError(
            f"Requested {size} rows from a sequence of {len(rows)}."
        )

    if size == len(rows):
        return rows

    start = (len(rows) - size) // 2
    return rows.iloc[start:start + size].copy()


def sample_not_ready(
    df: pd.DataFrame,
    dataset_name: str,
    target: int,
    min_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nir, candidates = choose_one_person_per_video(
        df=df,
        min_rows=min_rows,
        seed=seed,
    )

    whole, partial, remainder = choose_candidates_to_target(
        candidates=candidates,
        target=target,
    )

    selected_parts: list[pd.DataFrame] = []
    report: list[dict] = []

    for candidate in whole:
        rows = nir.loc[
            (nir["video_id"] == candidate.video_id)
            & (nir["person_id"] == candidate.person_id)
        ].copy()

        rows["selection_reason"] = (
            f"{dataset_name}_sampled_not_ready_whole_person"
        )
        selected_parts.append(rows)

        report.append(
            {
                "source_dataset": dataset_name,
                "video_id": candidate.video_id,
                "person_id": candidate.person_id,
                "available_not_ready_rows": candidate.num_rows,
                "selected_not_ready_rows": len(rows),
                "selection_type": "whole_person",
            }
        )

    if partial is not None:
        rows = nir.loc[
            (nir["video_id"] == partial.video_id)
            & (nir["person_id"] == partial.person_id)
        ].copy()

        rows = centered_segment(rows, remainder)
        rows["selection_reason"] = (
            f"{dataset_name}_sampled_not_ready_partial_person"
        )
        selected_parts.append(rows)

        report.append(
            {
                "source_dataset": dataset_name,
                "video_id": partial.video_id,
                "person_id": partial.person_id,
                "available_not_ready_rows": partial.num_rows,
                "selected_not_ready_rows": len(rows),
                "selection_type": "contiguous_partial_person",
            }
        )

    selected = pd.concat(selected_parts, ignore_index=True)

    if len(selected) != target:
        raise RuntimeError(
            f"{dataset_name}: selected {len(selected)} rows, "
            f"expected {target}."
        )

    report_df = pd.DataFrame(report)

    if report_df["video_id"].duplicated().any():
        duplicates = report_df.loc[
            report_df["video_id"].duplicated(keep=False),
            "video_id",
        ].unique()
        raise RuntimeError(
            f"{dataset_name}: more than one sampled person was selected "
            f"from videos {duplicates.tolist()}."
        )

    return selected, report_df


# ==============================================================================
# STATISTICS
# ==============================================================================

def unique_frames(df: pd.DataFrame) -> int:
    return int(
        df[
            ["source_dataset", "video_id", "frame_id"]
        ].drop_duplicates().shape[0]
    )


def unique_people(df: pd.DataFrame) -> int:
    return int(
        df[
            ["source_dataset", "video_id", "person_id"]
        ].drop_duplicates().shape[0]
    )


def label_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label, group in df.groupby("label", sort=True):
        rows.append(
            {
                "label": label,
                "num_person_frame_rows": len(group),
                "num_unique_image_frames": unique_frames(group),
                "num_unique_people": unique_people(group),
                "num_videos": group[
                    ["source_dataset", "video_id"]
                ].drop_duplicates().shape[0],
            }
        )

    return pd.DataFrame(rows)


def dataset_label_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (source, label), group in df.groupby(
        ["source_dataset", "label"],
        sort=True,
    ):
        rows.append(
            {
                "source_dataset": source,
                "label": label,
                "num_person_frame_rows": len(group),
                "num_unique_image_frames": unique_frames(group),
                "num_unique_people": unique_people(group),
                "num_videos": group["video_id"].nunique(),
            }
        )

    return pd.DataFrame(rows)


def dataset_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for source, group in df.groupby("source_dataset", sort=True):
        rows.append(
            {
                "source_dataset": source,
                "num_person_frame_rows": len(group),
                "num_unique_image_frames": unique_frames(group),
                "num_unique_people": unique_people(group),
                "num_videos": group["video_id"].nunique(),
                "not_interaction_ready_rows": int(
                    (group["label"] == LABEL_NOT_READY).sum()
                ),
                "interaction_ready_rows": int(
                    (group["label"] == LABEL_READY).sum()
                ),
                "interaction_ongoing_rows": int(
                    (group["label"] == LABEL_ONGOING).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def video_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (source, video_id), group in df.groupby(
        ["source_dataset", "video_id"],
        sort=True,
    ):
        rows.append(
            {
                "source_dataset": source,
                "video_id": video_id,
                "num_person_frame_rows": len(group),
                "num_unique_image_frames": group["frame_id"].nunique(),
                "num_people": group["person_id"].nunique(),
                "not_interaction_ready_rows": int(
                    (group["label"] == LABEL_NOT_READY).sum()
                ),
                "interaction_ready_rows": int(
                    (group["label"] == LABEL_READY).sum()
                ),
                "interaction_ongoing_rows": int(
                    (group["label"] == LABEL_ONGOING).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    if TARGET_EXTRA_NIR <= 0:
        raise ValueError("TARGET_EXTRA_NIR must be greater than zero.")

    if MIN_PERSON_FRAMES <= 0:
        raise ValueError("MIN_PERSON_FRAMES must be greater than zero.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading lightweight metadata...")
    jrdb = load_dataset_metadata(JRDB_PARQUET, "JRDB")
    avidar = load_dataset_metadata(AVIDAR_PARQUET, "AVIDAR")
    ssup = load_dataset_metadata(SSUP_PARQUET, "SSUP-HRI")

    print("Keeping all SSUP-HRI rows except interaction_done...")
    ssup_selected = ssup.loc[
        ssup["label"] != LABEL_DONE
    ].copy()
    ssup_selected["selection_reason"] = (
        "SSUP-HRI_all_except_interaction_done"
    )

    print("Keeping all AVIDAR interaction_ready rows...")
    avidar_ready = avidar.loc[
        avidar["label"] == LABEL_READY
    ].copy()
    avidar_ready["selection_reason"] = (
        "AVIDAR_all_interaction_ready"
    )

    print(
        f"Selecting {TARGET_EXTRA_NIR:,} not_interaction_ready "
        "rows from AVIDAR..."
    )
    avidar_nir, avidar_report = sample_not_ready(
        df=avidar,
        dataset_name="AVIDAR",
        target=TARGET_EXTRA_NIR,
        min_rows=MIN_PERSON_FRAMES,
        seed=RANDOM_SEED,
    )

    print(
        f"Selecting {TARGET_EXTRA_NIR:,} not_interaction_ready "
        "rows from JRDB..."
    )
    jrdb_nir, jrdb_report = sample_not_ready(
        df=jrdb,
        dataset_name="JRDB",
        target=TARGET_EXTRA_NIR,
        min_rows=MIN_PERSON_FRAMES,
        seed=RANDOM_SEED,
    )

    balanced = pd.concat(
        [
            ssup_selected,
            avidar_ready,
            avidar_nir,
            jrdb_nir,
        ],
        ignore_index=True,
    )

    unexpected_final_labels = sorted(
        set(balanced["label"].unique()) - FINAL_LABELS
    )
    if unexpected_final_labels:
        raise RuntimeError(
            "Unexpected final labels: "
            f"{unexpected_final_labels}"
        )

    # One exact manifest row per retained source-Parquet row.
    manifest = (
        balanced.sort_values(
            [
                "source_dataset",
                "video_id",
                "person_id",
                "_frame_sort",
                "source_row_index",
            ]
        )[
            [
                "source_dataset",
                "source_row_index",
                "video_id",
                "person_id",
                "frame_id",
                "label",
                "selection_reason",
            ]
        ]
        .reset_index(drop=True)
    )

    duplicate_rows = manifest.duplicated(
        ["source_dataset", "source_row_index"],
        keep=False,
    )
    if duplicate_rows.any():
        raise RuntimeError(
            "The manifest contains duplicate source rows."
        )

    selected_people = (
        pd.concat(
            [avidar_report, jrdb_report],
            ignore_index=True,
        )
        .sort_values(
            [
                "source_dataset",
                "selection_type",
                "selected_not_ready_rows",
                "video_id",
            ]
        )
        .reset_index(drop=True)
    )

    overall_by_label = label_statistics(balanced)
    by_dataset_and_label = dataset_label_statistics(balanced)
    by_dataset = dataset_statistics(balanced)
    by_video = video_statistics(balanced)

    manifest_path = OUTPUT_DIR / "balanced_selection.csv"
    selected_people_path = OUTPUT_DIR / "selected_people.csv"
    label_stats_path = OUTPUT_DIR / "label_statistics.csv"
    dataset_label_stats_path = (
        OUTPUT_DIR / "dataset_label_statistics.csv"
    )
    dataset_stats_path = OUTPUT_DIR / "dataset_statistics.csv"
    video_stats_path = OUTPUT_DIR / "video_statistics.csv"
    config_path = OUTPUT_DIR / "balancing_config.json"

    manifest.to_csv(manifest_path, index=False)
    selected_people.to_csv(selected_people_path, index=False)
    overall_by_label.to_csv(label_stats_path, index=False)
    by_dataset_and_label.to_csv(
        dataset_label_stats_path,
        index=False,
    )
    by_dataset.to_csv(dataset_stats_path, index=False)
    by_video.to_csv(video_stats_path, index=False)

    config = {
        "source_parquets": {
            "JRDB": str(JRDB_PARQUET),
            "AVIDAR": str(AVIDAR_PARQUET),
            "SSUP-HRI": str(SSUP_PARQUET),
        },
        "target_extra_not_interaction_ready_per_dataset": (
            TARGET_EXTRA_NIR
        ),
        "minimum_person_frames": MIN_PERSON_FRAMES,
        "random_seed": RANDOM_SEED,
        "exact_manifest_key": [
            "source_dataset",
            "source_row_index",
        ],
        "readable_identifiers": [
            "source_dataset",
            "video_id",
            "person_id",
            "frame_id",
        ],
        "rules": {
            "drop_interaction_done": True,
            "keep_all_ssup_except_interaction_done": True,
            "keep_all_avidar_interaction_ready": True,
            "maximum_sampled_people_per_video": 1,
            "partial_person_rows_are_listed_individually": True,
        },
    }

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\nOverall statistics per label:")
    print(overall_by_label.to_string(index=False))

    print("\nStatistics per dataset and label:")
    print(by_dataset_and_label.to_string(index=False))

    print("\nStatistics per dataset:")
    print(by_dataset.to_string(index=False))

    print("\nSaved:")
    print(f"  {manifest_path}")
    print(f"  {selected_people_path}")
    print(f"  {label_stats_path}")
    print(f"  {dataset_label_stats_path}")
    print(f"  {dataset_stats_path}")
    print(f"  {video_stats_path}")
    print(f"  {config_path}")
    print(f"\nSelected manifest rows: {len(manifest):,}")


if __name__ == "__main__":
    main()
