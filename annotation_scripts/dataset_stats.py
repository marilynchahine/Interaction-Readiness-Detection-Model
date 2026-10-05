from pathlib import Path
import json
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

JSON_DIR = Path(
    r"D:\EngagementDetection\data\JSON_hierarchical\full_dataset"
)

OUTPUT_DIR = Path(r"D:\EngagementDetection\data\data_stats\full_dataset")

# interaction_done is treated as not_interaction_ready.
LABEL_MAPPING = {
    # "interaction_done": "not_interaction_ready",
}

# Preferred order in tables and plots.
LABEL_ORDER = [
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
    "interaction_done"
]


# Frame-level aggregation:
# - interaction_ready person  -> interaction_ready frame
# - interaction_ongoing person -> interaction_ongoing frame
# - a frame is not_interaction_ready only when no person is ready/ongoing
FRAME_LABEL_ORDER = [
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
    "interaction_done"
]

FRAME_PERSON_LABEL_MAPPING = {
    "not_interaction_ready": "not_interaction_ready",
    "interaction_ready": "interaction_ready",
    "interaction_ongoing": "interaction_ongoing",
    "interaction_done": "interaction_done",
}

# Set to True to display plots while running the script.
# Plots are saved regardless of this setting.
SHOW_PLOTS = False

# Plot resolution.
PLOT_DPI = 200


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_label(label):
    """
    Normalize a raw annotation label.

    For example:
        interaction_done -> not_interaction_ready
    """
    if label is None:
        label = "unknown"

    label = str(label).strip()

    if not label:
        label = "unknown"

    return LABEL_MAPPING.get(label, label)


def safe_filename(name):
    """
    Convert a video ID into a filename-safe string.
    """
    name = str(name)
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def sort_person_id(person_id):
    """
    Return a sorting key that sorts numeric person IDs numerically.

    Examples:
        person_2 comes before person_10
        2 comes before 10

    Non-numeric IDs are sorted alphabetically after numeric IDs.
    """
    person_id = str(person_id)

    match = re.search(r"(\d+)$", person_id)

    if match:
        return 0, int(match.group(1)), person_id

    return 1, 0, person_id.lower()


def get_ordered_labels(discovered_labels):
    """
    Put the expected labels first, followed by any unexpected labels.
    """
    discovered_labels = set(discovered_labels)

    ordered = [
        label
        for label in LABEL_ORDER
        if label in discovered_labels
    ]

    extra_labels = sorted(discovered_labels - set(LABEL_ORDER))

    return ordered + extra_labels


def get_unique_frame_count(annotations):
    """
    Count unique frame IDs when frame_id is available.

    If an annotation does not have a frame_id, it is counted as a
    separate annotation.

    This prevents duplicate annotations for the same person and frame
    from incorrectly increasing the number of frames.
    """
    frame_keys = set()

    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            frame_keys.add(("annotation_index", index))
            continue

        frame_id = annotation.get("frame_id")

        if frame_id is None:
            frame_keys.add(("annotation_index", index))
        else:
            frame_keys.add(("frame_id", str(frame_id)))

    return len(frame_keys)


def count_frames_by_label(annotations):
    """
    Count unique frames for each label.

    The expected data structure is one annotation per person per frame.
    This function also protects against accidental duplicate entries
    with the same frame_id and label.
    """
    label_to_frame_keys = defaultdict(set)

    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            label = "unknown"
            frame_key = ("annotation_index", index)
        else:
            label = normalize_label(annotation.get("label", "unknown"))
            frame_id = annotation.get("frame_id")

            if frame_id is None:
                frame_key = ("annotation_index", index)
            else:
                frame_key = ("frame_id", str(frame_id))

        label_to_frame_keys[label].add(frame_key)

    return Counter({
        label: len(frame_keys)
        for label, frame_keys in label_to_frame_keys.items()
    })



def aggregate_frame_labels(data):
    """
    Return one label per unique video frame after combining all people.

    A frame is interaction_ready when at least one person is labeled
    interaction_ready or interaction_ongoing. Otherwise it is
    not_interaction_ready. interaction_done is treated as not ready.
    """
    labels_by_frame = defaultdict(set)

    for person_id, annotations in data.items():
        if person_id == "video_id" or not isinstance(annotations, list):
            continue

        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                frame_key = (
                    "missing_frame",
                    str(person_id),
                    annotation_index,
                )
                raw_label = "unknown"
            else:
                frame_id = annotation.get("frame_id")
                frame_key = (
                    ("frame_id", str(frame_id))
                    if frame_id is not None
                    else (
                        "missing_frame",
                        str(person_id),
                        annotation_index,
                    )
                )
                raw_label = normalize_label(
                    annotation.get("label", "unknown")
                )

            mapped_label = FRAME_PERSON_LABEL_MAPPING.get(
                raw_label,
                raw_label,
            )
            labels_by_frame[frame_key].add(mapped_label)

    frame_labels = {}

    for frame_key, labels in labels_by_frame.items():
        if "interaction_ready" in labels:
            frame_labels[frame_key] = "interaction_ready"
        elif "interaction_ongoing" in labels:
            frame_labels[frame_key] = "interaction_ongoing"
        elif "interaction_done" in labels:
            frame_labels[frame_key] = "interaction_done"
        elif labels == {"not_interaction_ready"}:
            frame_labels[frame_key] = "not_interaction_ready"
        else:
            frame_labels[frame_key] = "unknown"

    return frame_labels



def save_stacked_bar_plot(
    dataframe,
    index_column,
    label_columns,
    title,
    x_label,
    output_path,
):
    """
    Save a stacked bar plot from a statistics DataFrame.
    """
    if dataframe.empty:
        print(f"Skipping empty plot: {title}")
        return

    if not label_columns:
        print(f"Skipping plot with no label columns: {title}")
        return

    plot_data = dataframe.set_index(index_column)[label_columns]

    # Scale the figure width according to the number of bars.
    figure_width = max(10, len(plot_data) * 0.65)

    ax = plot_data.plot(
        kind="bar",
        stacked=True,
        figsize=(figure_width, 7),
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of frames")
    ax.legend(
        title="Interaction-readiness class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plots_directory = OUTPUT_DIR / "video_plots"
    plots_directory.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(JSON_DIR.glob("*.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files were found in:\n{JSON_DIR}"
        )

    # --------------------------------------------------------
    # Dataset-level accumulators
    # --------------------------------------------------------

    total_people = 0
    total_person_frames = 0
    frames_per_person = []

    dataset_label_frame_counts = Counter()
    dataset_label_people = defaultdict(set)
    dataset_label_videos = defaultdict(set)

    # Counts of unique video frames after combining all people in each frame.
    dataset_frame_level_counts = Counter()
    total_unique_video_frames = 0

    discovered_labels = set()

    # --------------------------------------------------------
    # Output records
    # --------------------------------------------------------

    person_records = []
    video_records = []
    frame_level_video_records = []

    # Used for the detailed JSON output.
    detailed_statistics = {
        "dataset": {},
        "videos": {},
    }

    # --------------------------------------------------------
    # Process each video
    # --------------------------------------------------------

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        video_id = str(data.get("video_id", json_path.stem))

        video_label_frame_counts = Counter()
        video_label_people = defaultdict(set)

        frame_labels = aggregate_frame_labels(data)
        video_frame_level_counts = Counter(frame_labels.values())

        total_unique_video_frames += len(frame_labels)
        dataset_frame_level_counts.update(video_frame_level_counts)

        frame_level_video_records.append(
            {
                "video_id": video_id,
                "source_file": json_path.name,
                "total_unique_frames": len(frame_labels),
                "not_interaction_ready": (
                    video_frame_level_counts.get(
                        "not_interaction_ready",
                        0,
                    )
                ),
                "interaction_ready": (
                    video_frame_level_counts.get(
                        "interaction_ready",
                        0,
                    )
                ),
                "unknown": video_frame_level_counts.get("unknown", 0),
            }
        )

        video_person_records = []
        video_total_person_frames = 0
        video_people_count = 0

        detailed_statistics["videos"][video_id] = {
            "source_file": json_path.name,
            "total_people": 0,
            "total_person_frames": 0,
            "frames_per_label": {},
            "people_per_label": {},
            "people": {},
        }

        for person_id, annotations in data.items():
            if person_id == "video_id":
                continue

            if not isinstance(annotations, list):
                continue

            person_id = str(person_id)
            person_global_id = f"{video_id}_{person_id}"

            person_label_counts = count_frames_by_label(annotations)
            person_total_frames = get_unique_frame_count(annotations)

            total_people += 1
            video_people_count += 1

            total_person_frames += person_total_frames
            video_total_person_frames += person_total_frames

            frames_per_person.append(person_total_frames)

            labels_for_person = {
                label
                for label, count in person_label_counts.items()
                if count > 0
            }

            for label, count in person_label_counts.items():
                discovered_labels.add(label)

                dataset_label_frame_counts[label] += count
                video_label_frame_counts[label] += count

            for label in labels_for_person:
                dataset_label_people[label].add(person_global_id)
                dataset_label_videos[label].add(video_id)
                video_label_people[label].add(person_id)

            person_record = {
                "video_id": video_id,
                "person_id": person_id,
                "person_global_id": person_global_id,
                "total_frames": person_total_frames,
            }

            for label, count in person_label_counts.items():
                person_record[label] = count

            person_records.append(person_record)
            video_person_records.append(person_record.copy())

            detailed_statistics["videos"][video_id]["people"][person_id] = {
                "total_frames": person_total_frames,
                "frames_per_label": dict(person_label_counts),
            }

        video_record = {
            "video_id": video_id,
            "source_file": json_path.name,
            "total_people": video_people_count,
            "total_person_frames": video_total_person_frames,
            "average_frames_per_person": (
                video_total_person_frames / video_people_count
                if video_people_count > 0
                else 0
            ),
        }

        for label, count in video_label_frame_counts.items():
            video_record[label] = count
            video_record[f"people_with_{label}"] = len(
                video_label_people[label]
            )

        video_records.append(video_record)

        detailed_statistics["videos"][video_id].update({
            "total_people": video_people_count,
            "total_person_frames": video_total_person_frames,
            "average_frames_per_person": (
                video_total_person_frames / video_people_count
                if video_people_count > 0
                else 0
            ),
            "frames_per_label": dict(video_label_frame_counts),
            "people_per_label": {
                label: len(people)
                for label, people in video_label_people.items()
            },
        })

    # ========================================================
    # FINALIZE TABLES
    # ========================================================

    ordered_labels = get_ordered_labels(discovered_labels)

    person_df = pd.DataFrame(person_records)
    video_df = pd.DataFrame(video_records)
    frame_level_video_df = pd.DataFrame(frame_level_video_records)

    # Add missing label columns with zeros.
    for label in ordered_labels:
        if label not in person_df.columns:
            person_df[label] = 0

        if label not in video_df.columns:
            video_df[label] = 0

        people_column = f"people_with_{label}"

        if people_column not in video_df.columns:
            video_df[people_column] = 0

    # Ensure frame-count columns are integers.
    for label in ordered_labels:
        person_df[label] = person_df[label].fillna(0).astype(int)
        video_df[label] = video_df[label].fillna(0).astype(int)

        people_column = f"people_with_{label}"
        video_df[people_column] = (
            video_df[people_column]
            .fillna(0)
            .astype(int)
        )

    # Order people by video and person ID.
    if not person_df.empty:
        person_df["_person_sort_key"] = person_df["person_id"].apply(
            sort_person_id
        )

        person_df = (
            person_df
            .sort_values(
                by=["video_id", "_person_sort_key"],
                kind="stable",
            )
            .drop(columns="_person_sort_key")
            .reset_index(drop=True)
        )

    if not video_df.empty:
        video_df = (
            video_df
            .sort_values("video_id")
            .reset_index(drop=True)
        )


    if not frame_level_video_df.empty:
        frame_level_video_df = (
            frame_level_video_df
            .sort_values("video_id")
            .reset_index(drop=True)
        )

    # Reorder columns.
    person_columns = [
        "video_id",
        "person_id",
        "person_global_id",
        "total_frames",
        *ordered_labels,
    ]

    video_columns = [
        "video_id",
        "source_file",
        "total_people",
        "total_person_frames",
        "average_frames_per_person",
        *ordered_labels,
        *[
            f"people_with_{label}"
            for label in ordered_labels
        ],
    ]

    person_df = person_df[person_columns]
    video_df = video_df[video_columns]

    # ========================================================
    # DATASET-LEVEL STATISTICS
    # ========================================================

    average_frames_per_person = (
        total_person_frames / total_people
        if total_people > 0
        else 0
    )

    minimum_frames_per_person = (
        min(frames_per_person)
        if frames_per_person
        else 0
    )

    maximum_frames_per_person = (
        max(frames_per_person)
        if frames_per_person
        else 0
    )

    median_frames_per_person = (
        float(pd.Series(frames_per_person).median())
        if frames_per_person
        else 0
    )

    dataset_summary = {
        "total_videos": len(json_paths),
        "total_people": total_people,
        "total_person_frames": total_person_frames,
        "average_frames_per_person": average_frames_per_person,
        "median_frames_per_person": median_frames_per_person,
        "minimum_frames_per_person": minimum_frames_per_person,
        "maximum_frames_per_person": maximum_frames_per_person,
        "frames_per_label": {
            label: dataset_label_frame_counts.get(label, 0)
            for label in ordered_labels
        },
        "people_per_label": {
            label: len(dataset_label_people.get(label, set()))
            for label in ordered_labels
        },
        "videos_per_label": {
            label: len(dataset_label_videos.get(label, set()))
            for label in ordered_labels
        },
        "total_unique_video_frames": total_unique_video_frames,
        "aggregated_frames_per_label": {
            label: dataset_frame_level_counts.get(label, 0)
            for label in FRAME_LABEL_ORDER
        },
        "aggregated_unknown_frames": (
            dataset_frame_level_counts.get("unknown", 0)
        ),
    }

    detailed_statistics["dataset"] = dataset_summary

    # A tabular dataset summary.
    dataset_rows = [
        {
            "statistic": "total_videos",
            "value": len(json_paths),
        },
        {
            "statistic": "total_people",
            "value": total_people,
        },
        {
            "statistic": "total_person_frames",
            "value": total_person_frames,
        },
        {
            "statistic": "average_frames_per_person",
            "value": average_frames_per_person,
        },
        {
            "statistic": "median_frames_per_person",
            "value": median_frames_per_person,
        },
        {
            "statistic": "minimum_frames_per_person",
            "value": minimum_frames_per_person,
        },
        {
            "statistic": "maximum_frames_per_person",
            "value": maximum_frames_per_person,
        },
    ]

    for label in ordered_labels:
        dataset_rows.append({
            "statistic": f"frames_{label}",
            "value": dataset_label_frame_counts.get(label, 0),
        })

        dataset_rows.append({
            "statistic": f"people_with_{label}",
            "value": len(dataset_label_people.get(label, set())),
        })

        dataset_rows.append({
            "statistic": f"videos_with_{label}",
            "value": len(dataset_label_videos.get(label, set())),
        })

    dataset_rows.append({
        "statistic": "total_unique_video_frames",
        "value": total_unique_video_frames,
    })

    for frame_label in FRAME_LABEL_ORDER:
        dataset_rows.append({
            "statistic": f"aggregated_frames_{frame_label}",
            "value": dataset_frame_level_counts.get(frame_label, 0),
        })

    if dataset_frame_level_counts.get("unknown", 0) > 0:
        dataset_rows.append({
            "statistic": "aggregated_frames_unknown",
            "value": dataset_frame_level_counts.get("unknown", 0),
        })

    dataset_df = pd.DataFrame(dataset_rows)

    # ========================================================
    # SAVE TABLES
    # ========================================================

    dataset_csv_path = OUTPUT_DIR / "dataset_statistics.csv"
    person_csv_path = OUTPUT_DIR / "person_statistics.csv"
    video_csv_path = OUTPUT_DIR / "video_statistics.csv"
    frame_level_video_csv_path = (
        OUTPUT_DIR / "frame_level_video_statistics.csv"
    )
    frame_level_dataset_csv_path = (
        OUTPUT_DIR / "frame_level_dataset_statistics.csv"
    )
    detailed_json_path = OUTPUT_DIR / "detailed_statistics.json"

    frame_level_dataset_df = pd.DataFrame(
        [
            {
                "frame_label": label,
                "num_unique_frames": dataset_frame_level_counts.get(
                    label,
                    0,
                ),
                "percentage": (
                    dataset_frame_level_counts.get(label, 0)
                    / total_unique_video_frames
                    * 100
                    if total_unique_video_frames > 0
                    else 0
                ),
            }
            for label in FRAME_LABEL_ORDER
        ]
    )

    dataset_df.to_csv(dataset_csv_path, index=False)
    person_df.to_csv(person_csv_path, index=False)
    video_df.to_csv(video_csv_path, index=False)
    frame_level_video_df.to_csv(
        frame_level_video_csv_path,
        index=False,
    )
    frame_level_dataset_df.to_csv(
        frame_level_dataset_csv_path,
        index=False,
    )

    with open(detailed_json_path, "w", encoding="utf-8") as file:
        json.dump(
            detailed_statistics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # ========================================================
    # CREATE ONE STACKED BAR PLOT PER VIDEO
    # ========================================================

    for video_id in video_df["video_id"]:
        current_video_df = person_df[
            person_df["video_id"] == video_id
        ].copy()

        if current_video_df.empty:
            continue

        current_video_df["_person_sort_key"] = (
            current_video_df["person_id"].apply(sort_person_id)
        )

        current_video_df = (
            current_video_df
            .sort_values("_person_sort_key", kind="stable")
            .drop(columns="_person_sort_key")
        )

        plot_path = (
            plots_directory
            / f"{safe_filename(video_id)}_person_class_counts.png"
        )

        save_stacked_bar_plot(
            dataframe=current_video_df,
            index_column="person_id",
            label_columns=ordered_labels,
            title=(
                f"Interaction-readiness frame counts per person\n"
                f"Video: {video_id}"
            ),
            x_label="Person ID",
            output_path=plot_path,
        )

    # ========================================================
    # CREATE AGGREGATE VIDEO COMPARISON PLOT
    # ========================================================

    aggregate_plot_path = (
        OUTPUT_DIR / "all_videos_class_counts.png"
    )

    save_stacked_bar_plot(
        dataframe=video_df,
        index_column="video_id",
        label_columns=ordered_labels,
        title="Interaction-readiness frame counts per video",
        x_label="Video ID",
        output_path=aggregate_plot_path,
    )

    # ========================================================
    # PRINT DATASET STATISTICS
    # ========================================================

    print("\n========== DATASET STATISTICS ==========")
    print(f"Total videos: {len(json_paths)}")
    print(f"Total people: {total_people}")
    print(f"Total person-frame annotations: {total_person_frames}")
    print(
        f"Average frames per person: "
        f"{average_frames_per_person:.2f}"
    )
    print(
        f"Median frames per person: "
        f"{median_frames_per_person:.2f}"
    )
    print(f"Minimum frames per person: {minimum_frames_per_person}")
    print(f"Maximum frames per person: {maximum_frames_per_person}")

    print("\nFrames per label:")
    for label in ordered_labels:
        count = dataset_label_frame_counts.get(label, 0)
        percentage = (
            count / total_person_frames * 100
            if total_person_frames > 0
            else 0
        )

        print(
            f"  {label}: {count} "
            f"({percentage:.2f}%)"
        )

    print("\nUnique video frames after combining all people:")
    print(f"  Total unique video frames: {total_unique_video_frames}")

    for frame_label in FRAME_LABEL_ORDER:
        count = dataset_frame_level_counts.get(frame_label, 0)
        percentage = (
            count / total_unique_video_frames * 100
            if total_unique_video_frames > 0
            else 0
        )
        print(
            f"  {frame_label}: {count} "
            f"({percentage:.2f}%)"
        )

    if dataset_frame_level_counts.get("unknown", 0) > 0:
        print(
            "  unknown: "
            f"{dataset_frame_level_counts.get('unknown', 0)}"
        )

    print("\nPeople per label:")
    for label in ordered_labels:
        people_count = len(
            dataset_label_people.get(label, set())
        )

        print(f"  {label}: {people_count}")

    print("\nVideos per label:")
    for label in ordered_labels:
        videos_count = len(
            dataset_label_videos.get(label, set())
        )

        print(f"  {label}: {videos_count}")

    print("========================================")

    # ========================================================
    # PRINT VIDEO-BY-VIDEO STATISTICS
    # ========================================================

    print("\n========== VIDEO STATISTICS ==========")

    for _, video_row in video_df.iterrows():
        print(f"\nVideo: {video_row['video_id']}")
        print(f"  People: {video_row['total_people']}")
        print(
            f"  Person-frame annotations: "
            f"{video_row['total_person_frames']}"
        )
        print(
            f"  Average frames per person: "
            f"{video_row['average_frames_per_person']:.2f}"
        )

        print("  Frames per label:")

        for label in ordered_labels:
            print(
                f"    {label}: "
                f"{int(video_row[label])}"
            )

    print("\n======================================")

    # ========================================================
    # PRINT PERSON-BY-PERSON STATISTICS
    # ========================================================

    print("\n========== PERSON STATISTICS ==========")

    for video_id, group in person_df.groupby(
        "video_id",
        sort=True,
    ):
        print(f"\nVideo: {video_id}")

        for _, person_row in group.iterrows():
            print(
                f"  Person {person_row['person_id']}: "
                f"{person_row['total_frames']} total frames"
            )

            for label in ordered_labels:
                print(
                    f"    {label}: "
                    f"{int(person_row[label])}"
                )

    print("\n=======================================")

    # ========================================================
    # PRINT OUTPUT LOCATIONS
    # ========================================================

    print("\nSaved outputs:")
    print(f"  Dataset statistics: {dataset_csv_path}")
    print(f"  Person statistics: {person_csv_path}")
    print(f"  Video statistics: {video_csv_path}")
    print(
        f"  Frame-level dataset statistics: "
        f"{frame_level_dataset_csv_path}"
    )
    print(
        f"  Frame-level video statistics: "
        f"{frame_level_video_csv_path}"
    )
    print(f"  Detailed JSON: {detailed_json_path}")
    print(f"  Video plots: {plots_directory}")
    print(f"  Aggregate plot: {aggregate_plot_path}")


if __name__ == "__main__":
    main()