#!/usr/bin/env python3
"""
Evaluate VIDEO-LEVEL Qwen interaction-readiness predictions.

This evaluator is the video-level equivalent of the frame-level evaluator.
It evaluates ONLY the target frame (frame 16) prediction stored in each
temporal-window record.

Expected input
--------------
A predictions.jsonl produced by Qwen_videoLevel_ZeroShot_predict.py.

Each line contains:

{
    "source_dataset": "...",
    "video_id": "...",
    "frame_id": ...,
    "frame_group": ...,
    "ground_truth_people": [
        {
            "person_id": ...,
            "bbox_xyxy": [x1, y1, x2, y2],
            "label_original": "...",
            "label_3class": "..."
        }
    ],
    "predicted_people": [
        {
            "person_id": ...,
            "bbox_2d": [x1, y1, x2, y2],
            "label": "..."
        }
    ],
    "image_error": null,
    "parse_error": null,
    ...
}

Evaluation logic
----------------
1. For each temporal group, evaluate its TARGET frame only.
2. Build the full pairwise IoU matrix between target-frame GT boxes and
   predicted boxes.
3. Save every pairwise IoU BEFORE Hungarian assignment.
4. Use Hungarian assignment to maximize total IoU.
5. Accept assigned pairs only when IoU >= IOU_THRESHOLD.
6. Accepted pairs are detection true positives.
7. Unmatched predictions are false positives.
8. Unmatched ground-truth boxes are false negatives.
9. Classification metrics are computed ONLY on accepted detection matches.

Detection metrics
-----------------
- Mean IoU over true-positive matches
- Median IoU over true-positive matches
- Detection precision
- Detection recall
- Detection F1-score
- TP / FP / FN counts
- Number of GT and predicted boxes
- Number of image / parse errors

Classification metrics
----------------------
- Accuracy
- Per-class precision
- Per-class recall
- Per-class F1-score
- Macro F1
- Weighted F1
- Confusion matrix

Outputs
-------
evaluation_summary.json
detection_metrics.csv
classification_metrics.csv
confusion_matrix.csv
confusion_matrix.png
matched_predictions.csv
frame_detection_details.csv
all_iou_pairs_before_hungarian.csv

In addition, the same complete evaluation is produced separately for every
source_dataset under:

by_dataset/<dataset_name>/

A compact cross-dataset comparison is also saved as:
dataset_comparison.csv

Required packages
-----------------
pip install numpy pandas scipy scikit-learn matplotlib
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ==============================================================================
# CONFIGURATION — EDIT THESE VALUES
# ==============================================================================

PREDICTIONS_JSONL = Path(
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/videoLevel_fineTuned/Qwen3VL_2B_Instruct/inference_RollOut/predictions.jsonl"
)

OUTPUT_DIR = Path(
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/videoLevel_fineTuned/Qwen3VL_2B_Instruct/inference_RollOut/evaluation"
)

# A predicted box is a true-positive detection only when IoU is at least this.
IOU_THRESHOLD = 0.50

CLASS_NAMES = [
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
]

# Kept for compatibility with old annotations.
MAP_INTERACTION_DONE_TO_NOT_READY = True

# only evaluate the top 5 people the model predicted
evaluate_top5 = True

# ==============================================================================
# Validation / normalization
# ==============================================================================

def normalize_label(label: Any) -> str:
    normalized = str(label).strip().lower()

    if (
        MAP_INTERACTION_DONE_TO_NOT_READY
        and normalized == "interaction_done"
    ):
        normalized = "not_interaction_ready"

    if normalized not in CLASS_NAMES:
        raise ValueError(
            f"Unsupported label {label!r}. Expected one of {CLASS_NAMES}."
        )

    return normalized


def normalize_box(box: Any, field_name: str) -> list[float]:
    if hasattr(box, "tolist"):
        box = box.tolist()

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(
            f"{field_name} must contain four coordinates, got: {box!r}"
        )

    x1, y1, x2, y2 = [float(value) for value in box]

    # Be conservative if coordinates were accidentally reversed.
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    if right <= left or bottom <= top:
        raise ValueError(
            f"{field_name} has zero or negative area: {box!r}"
        )

    return [left, top, right, bottom]


def load_prediction_records(path: Path, evaluate_top5) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            required = {
                "source_dataset",
                "video_id",
                "frame_id",
                "ground_truth_people",
                "predicted_people",
            }

            missing = required - set(record)
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: "
                    f"{sorted(missing)}"
                )

            if not isinstance(record["ground_truth_people"], list):
                raise ValueError(
                    f"Line {line_number}: "
                    "ground_truth_people must be a list."
                )

            if not isinstance(record["predicted_people"], list):
                raise ValueError(
                    f"Line {line_number}: predicted_people must be a list."
                )

            records.append(record)

    if not records:
        raise ValueError(f"No prediction records were found in {path}.")

    # Normally the video-level inference script rebuilds predictions.jsonl from
    # one per-group output file, so duplicates should not exist.
    #
    # Still, protect against accidental duplicates. frame_group is included
    # because temporal groups are the natural unit of this experiment.
    latest_by_group: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    
    total = 0
    for record in records:
        group_key = (
            str(record["source_dataset"]),
            str(record["video_id"]),
            str(record.get("frame_group", "")),
            str(record["frame_id"]),
        )

        if group_key in latest_by_group:
            duplicate_count += 1
        
        if evaluate_top5 & (record['predicted_people'] != None):
            record['predicted_people'] = record['predicted_people'][0:5]

        latest_by_group[group_key] = record

    if duplicate_count:
        print(
            f"Found {duplicate_count:,} duplicate temporal-group record(s); "
            "keeping only the latest copy of each group."
        )

    return list(latest_by_group.values())


# ==============================================================================
# IoU and Hungarian matching
# ==============================================================================

def box_iou(
    box_a: list[float],
    box_b: list[float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_left = max(ax1, bx1)
    intersection_top = max(ay1, by1)
    intersection_right = min(ax2, bx2)
    intersection_bottom = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_right - intersection_left,
    )
    intersection_height = max(
        0.0,
        intersection_bottom - intersection_top,
    )
    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def build_iou_matrix(
    ground_truth_boxes: list[list[float]],
    predicted_boxes: list[list[float]],
) -> np.ndarray:
    matrix = np.zeros(
        (
            len(ground_truth_boxes),
            len(predicted_boxes),
        ),
        dtype=np.float64,
    )

    for gt_index, gt_box in enumerate(
        ground_truth_boxes
    ):
        for pred_index, pred_box in enumerate(
            predicted_boxes
        ):
            matrix[gt_index, pred_index] = box_iou(
                gt_box,
                pred_box,
            )

    return matrix


def match_frame(
    ground_truth_people: list[dict[str, Any]],
    predicted_people: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    """
    Match target-frame predictions to ground truth using Hungarian assignment.

    Hungarian assignment maximizes total IoU globally. Assigned pairs below the
    threshold are rejected after assignment.
    """

    gt_boxes = [
        normalize_box(
            person["bbox_xyxy"],
            "ground-truth bbox_xyxy",
        )
        for person in ground_truth_people
    ]

    pred_boxes = [
        normalize_box(
            person["bbox_2d"],
            "predicted bbox_2d",
        )
        for person in predicted_people
    ]

    if not gt_boxes and not pred_boxes:
        return {
            "matches": [],
            "unmatched_gt_indices": [],
            "unmatched_pred_indices": [],
            "iou_matrix": np.empty(
                (0, 0),
                dtype=np.float64,
            ),
        }

    if not gt_boxes:
        return {
            "matches": [],
            "unmatched_gt_indices": [],
            "unmatched_pred_indices": list(
                range(len(pred_boxes))
            ),
            "iou_matrix": np.empty(
                (0, len(pred_boxes)),
                dtype=np.float64,
            ),
        }

    if not pred_boxes:
        return {
            "matches": [],
            "unmatched_gt_indices": list(
                range(len(gt_boxes))
            ),
            "unmatched_pred_indices": [],
            "iou_matrix": np.empty(
                (len(gt_boxes), 0),
                dtype=np.float64,
            ),
        }

    iou_matrix = build_iou_matrix(
        gt_boxes,
        pred_boxes,
    )

    # scipy minimizes cost, so minimize 1 - IoU.
    gt_indices, pred_indices = linear_sum_assignment(
        1.0 - iou_matrix
    )

    accepted_matches: list[dict[str, Any]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()

    for gt_index, pred_index in zip(
        gt_indices,
        pred_indices,
    ):
        iou = float(
            iou_matrix[gt_index, pred_index]
        )

        if iou >= iou_threshold:
            accepted_matches.append(
                {
                    "gt_index": int(gt_index),
                    "pred_index": int(pred_index),
                    "iou": iou,
                }
            )
            matched_gt.add(int(gt_index))
            matched_pred.add(int(pred_index))

    unmatched_gt = [
        index
        for index in range(len(gt_boxes))
        if index not in matched_gt
    ]

    unmatched_pred = [
        index
        for index in range(len(pred_boxes))
        if index not in matched_pred
    ]

    return {
        "matches": accepted_matches,
        "unmatched_gt_indices": unmatched_gt,
        "unmatched_pred_indices": unmatched_pred,
        "iou_matrix": iou_matrix,
    }


# ==============================================================================
# Metrics
# ==============================================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_records(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_ious: list[float] = []
    classification_true: list[str] = []
    classification_pred: list[str] = []

    matched_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    all_iou_rows: list[dict[str, Any]] = []

    frames_with_image_error = 0
    frames_with_parse_error = 0

    for record in records:
        source_dataset = str(
            record["source_dataset"]
        )
        video_id = str(
            record["video_id"]
        )
        frame_id = str(
            record["frame_id"]
        )
        frame_group = record.get(
            "frame_group"
        )
        split = record.get(
            "split"
        )

        if record.get("image_error"):
            frames_with_image_error += 1

        if record.get("parse_error"):
            frames_with_parse_error += 1

        ground_truth_people = record[
            "ground_truth_people"
        ]
        predicted_people = record[
            "predicted_people"
        ]

        result = match_frame(
            ground_truth_people=ground_truth_people,
            predicted_people=predicted_people,
            iou_threshold=IOU_THRESHOLD,
        )

        # --------------------------------------------------------------
        # Save ALL GT x prediction IoUs BEFORE Hungarian assignment.
        # --------------------------------------------------------------
        iou_matrix = result["iou_matrix"]

        for gt_index, gt in enumerate(
            ground_truth_people
        ):
            for pred_index, pred in enumerate(
                predicted_people
            ):
                all_iou_rows.append(
                    {
                        "source_dataset": source_dataset,
                        "video_id": video_id,
                        "frame_group": frame_group,
                        "target_frame_id": frame_id,
                        "split": split,
                        "gt_index": gt_index,
                        "pred_index": pred_index,
                        "gt_person_id": gt.get(
                            "person_id"
                        ),
                        "pred_person_id": pred.get(
                            "person_id"
                        ),
                        "gt_bbox_xyxy": json.dumps(
                            gt["bbox_xyxy"]
                        ),
                        "pred_bbox_xyxy": json.dumps(
                            pred["bbox_2d"]
                        ),
                        "gt_label": normalize_label(
                            gt.get(
                                "label_3class",
                                gt.get("label_original"),
                            )
                        ),
                        "pred_label": normalize_label(
                            pred["label"]
                        ),
                        "iou": float(
                            iou_matrix[
                                gt_index,
                                pred_index,
                            ]
                        ),
                    }
                )

        frame_tp = len(
            result["matches"]
        )
        frame_fn = len(
            result["unmatched_gt_indices"]
        )
        frame_fp = len(
            result["unmatched_pred_indices"]
        )

        true_positives += frame_tp
        false_negatives += frame_fn
        false_positives += frame_fp

        # --------------------------------------------------------------
        # Classification is evaluated ONLY on accepted detection matches.
        # --------------------------------------------------------------
        for match in result["matches"]:
            gt = ground_truth_people[
                match["gt_index"]
            ]
            pred = predicted_people[
                match["pred_index"]
            ]

            gt_label = normalize_label(
                gt.get(
                    "label_3class",
                    gt.get("label_original"),
                )
            )
            pred_label = normalize_label(
                pred["label"]
            )
            iou = float(
                match["iou"]
            )

            classification_true.append(
                gt_label
            )
            classification_pred.append(
                pred_label
            )
            matched_ious.append(
                iou
            )

            matched_rows.append(
                {
                    "source_dataset": source_dataset,
                    "video_id": video_id,
                    "frame_group": frame_group,
                    "target_frame_id": frame_id,
                    "split": split,
                    "gt_person_id": gt.get(
                        "person_id"
                    ),
                    "pred_person_id": pred.get(
                        "person_id"
                    ),
                    "gt_bbox_xyxy": json.dumps(
                        gt["bbox_xyxy"]
                    ),
                    "pred_bbox_xyxy": json.dumps(
                        pred["bbox_2d"]
                    ),
                    "iou": iou,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "classification_correct": (
                        gt_label == pred_label
                    ),
                }
            )

        frame_rows.append(
            {
                "source_dataset": source_dataset,
                "video_id": video_id,
                "frame_group": frame_group,
                "target_frame_id": frame_id,
                "split": split,
                "num_ground_truth_people": len(
                    ground_truth_people
                ),
                "num_predicted_people": len(
                    predicted_people
                ),
                "true_positives": frame_tp,
                "false_positives": frame_fp,
                "false_negatives": frame_fn,
                "mean_matched_iou": (
                    float(
                        np.mean(
                            [
                                match["iou"]
                                for match
                                in result["matches"]
                            ]
                        )
                    )
                    if result["matches"]
                    else np.nan
                ),
                "image_error": record.get(
                    "image_error"
                ),
                "parse_error": record.get(
                    "parse_error"
                ),
            }
        )

    # ------------------------------------------------------------------
    # Detection metrics
    # ------------------------------------------------------------------
    detection_precision = safe_divide(
        true_positives,
        true_positives + false_positives,
    )
    detection_recall = safe_divide(
        true_positives,
        true_positives + false_negatives,
    )
    detection_f1 = safe_divide(
        2
        * detection_precision
        * detection_recall,
        detection_precision
        + detection_recall,
    )

    detection_metrics = {
        "iou_threshold": IOU_THRESHOLD,
        "num_target_frames": len(records),
        "num_ground_truth_boxes": (
            true_positives + false_negatives
        ),
        "num_predicted_boxes": (
            true_positives + false_positives
        ),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "mean_iou_true_positive_matches": (
            float(np.mean(matched_ious))
            if matched_ious
            else 0.0
        ),
        "median_iou_true_positive_matches": (
            float(np.median(matched_ious))
            if matched_ious
            else 0.0
        ),
        "detection_precision": float(
            detection_precision
        ),
        "detection_recall": float(
            detection_recall
        ),
        "detection_f1": float(
            detection_f1
        ),
        "frames_with_image_error": (
            frames_with_image_error
        ),
        "frames_with_parse_error": (
            frames_with_parse_error
        ),
    }

    # ------------------------------------------------------------------
    # Classification metrics
    # ------------------------------------------------------------------
    if classification_true:
        classification_accuracy = accuracy_score(
            classification_true,
            classification_pred,
        )

        macro_f1 = f1_score(
            classification_true,
            classification_pred,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            classification_true,
            classification_pred,
            labels=CLASS_NAMES,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            classification_true,
            classification_pred,
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            classification_true,
            classification_pred,
            labels=CLASS_NAMES,
        )

    else:
        classification_accuracy = 0.0
        macro_f1 = 0.0
        weighted_f1 = 0.0

        report = {
            class_name: {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 0.0,
            }
            for class_name in CLASS_NAMES
        }

        matrix = np.zeros(
            (
                len(CLASS_NAMES),
                len(CLASS_NAMES),
            ),
            dtype=int,
        )

    per_class_metrics: list[dict[str, Any]] = []

    for class_name in CLASS_NAMES:
        class_result = report[
            class_name
        ]

        per_class_metrics.append(
            {
                "class": class_name,
                "precision": float(
                    class_result["precision"]
                ),
                "recall": float(
                    class_result["recall"]
                ),
                "f1_score": float(
                    class_result["f1-score"]
                ),
                "support": int(
                    class_result["support"]
                ),
            }
        )

    classification_metrics = {
        "num_matched_detections_used_for_classification": len(
            classification_true
        ),
        "accuracy": float(
            classification_accuracy
        ),
        "macro_f1": float(
            macro_f1
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "per_class": per_class_metrics,
        "confusion_matrix_labels": CLASS_NAMES,
        "confusion_matrix": matrix.tolist(),
    }

    summary = {
        "detection": detection_metrics,
        "classification": classification_metrics,
    }

    return (
        summary,
        pd.DataFrame(matched_rows),
        pd.DataFrame(frame_rows),
        pd.DataFrame(all_iou_rows),
    )


# ==============================================================================
# Saving
# ==============================================================================

def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
    )
    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set(
        xticks=np.arange(
            len(CLASS_NAMES)
        ),
        yticks=np.arange(
            len(CLASS_NAMES)
        ),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="Ground-truth label",
        title=(
            "Interaction Readiness "
            "Confusion Matrix — Video Level"
        ),
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size
        and matrix.max() > 0
        else 0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index,
                    ]
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[
                        row_index,
                        column_index,
                    ]
                    > threshold
                    else "black"
                ),
            )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(
        figure
    )


def build_metric_tables(
    summary: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the same metric tables used for the overall and per-dataset runs."""

    detection_df = pd.DataFrame(
        [
            {
                key: value
                for key, value
                in summary["detection"].items()
            }
        ]
    )

    classification_df = pd.DataFrame(
        summary["classification"]["per_class"]
    )

    classification_df["accuracy"] = np.nan
    classification_df["macro_f1"] = np.nan
    classification_df["weighted_f1"] = np.nan

    overall_classification_row = pd.DataFrame(
        [
            {
                "class": "OVERALL",
                "precision": np.nan,
                "recall": np.nan,
                "f1_score": np.nan,
                "support": summary[
                    "classification"
                ][
                    "num_matched_detections_used_for_classification"
                ],
                "accuracy": summary[
                    "classification"
                ]["accuracy"],
                "macro_f1": summary[
                    "classification"
                ]["macro_f1"],
                "weighted_f1": summary[
                    "classification"
                ]["weighted_f1"],
            }
        ]
    )

    classification_df = pd.concat(
        [
            classification_df,
            overall_classification_row,
        ],
        ignore_index=True,
    )

    confusion_df = pd.DataFrame(
        summary["classification"]["confusion_matrix"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    confusion_df.index.name = "ground_truth"
    confusion_df.columns.name = "predicted"

    return (
        detection_df,
        classification_df,
        confusion_df,
    )


def save_evaluation_bundle(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Run the complete evaluation and save exactly the same outputs for a subset
    of records. Used once globally and once for each source dataset.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        summary,
        matched_df,
        frame_df,
        all_iou_df,
    ) = evaluate_records(
        records
    )

    (
        detection_df,
        classification_df,
        confusion_df,
    ) = build_metric_tables(
        summary
    )

    summary_path = (
        output_dir
        / "evaluation_summary.json"
    )
    detection_path = (
        output_dir
        / "detection_metrics.csv"
    )
    classification_path = (
        output_dir
        / "classification_metrics.csv"
    )
    confusion_csv_path = (
        output_dir
        / "confusion_matrix.csv"
    )
    confusion_png_path = (
        output_dir
        / "confusion_matrix.png"
    )
    matched_path = (
        output_dir
        / "matched_predictions.csv"
    )
    frame_path = (
        output_dir
        / "frame_detection_details.csv"
    )
    all_iou_path = (
        output_dir
        / "all_iou_pairs_before_hungarian.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    detection_df.to_csv(
        detection_path,
        index=False,
    )
    classification_df.to_csv(
        classification_path,
        index=False,
    )
    confusion_df.to_csv(
        confusion_csv_path
    )
    matched_df.to_csv(
        matched_path,
        index=False,
    )
    frame_df.to_csv(
        frame_path,
        index=False,
    )
    all_iou_df.to_csv(
        all_iou_path,
        index=False,
    )

    save_confusion_matrix(
        matrix=np.asarray(
            summary[
                "classification"
            ]["confusion_matrix"],
            dtype=int,
        ),
        output_path=confusion_png_path,
    )

    return summary


def make_dataset_comparison_row(
    dataset_name: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Flatten the key overall + per-class metrics into one row so datasets can be
    compared directly in a single CSV.
    """
    detection = summary["detection"]
    classification = summary["classification"]

    row: dict[str, Any] = {
        "source_dataset": dataset_name,

        "num_target_frames": detection[
            "num_target_frames"
        ],
        "num_ground_truth_boxes": detection[
            "num_ground_truth_boxes"
        ],
        "num_predicted_boxes": detection[
            "num_predicted_boxes"
        ],

        "true_positives": detection[
            "true_positives"
        ],
        "false_positives": detection[
            "false_positives"
        ],
        "false_negatives": detection[
            "false_negatives"
        ],

        "mean_iou": detection[
            "mean_iou_true_positive_matches"
        ],
        "median_iou": detection[
            "median_iou_true_positive_matches"
        ],
        "detection_precision": detection[
            "detection_precision"
        ],
        "detection_recall": detection[
            "detection_recall"
        ],
        "detection_f1": detection[
            "detection_f1"
        ],

        "frames_with_image_error": detection[
            "frames_with_image_error"
        ],
        "frames_with_parse_error": detection[
            "frames_with_parse_error"
        ],

        "num_matched_for_classification": classification[
            "num_matched_detections_used_for_classification"
        ],
        "classification_accuracy": classification[
            "accuracy"
        ],
        "classification_macro_f1": classification[
            "macro_f1"
        ],
        "classification_weighted_f1": classification[
            "weighted_f1"
        ],
    }

    for class_result in classification[
        "per_class"
    ]:
        class_name = class_result[
            "class"
        ]

        row[
            f"{class_name}_precision"
        ] = class_result[
            "precision"
        ]
        row[
            f"{class_name}_recall"
        ] = class_result[
            "recall"
        ]
        row[
            f"{class_name}_f1"
        ] = class_result[
            "f1_score"
        ]
        row[
            f"{class_name}_support"
        ] = class_result[
            "support"
        ]

    return row


def print_summary(
    title: str,
    summary: dict[str, Any],
) -> None:
    detection = summary["detection"]
    classification = summary["classification"]

    print(
        f"\n{'=' * 72}"
    )
    print(
        title
    )
    print(
        f"{'=' * 72}"
    )

    print(
        f"Target frames: {detection['num_target_frames']}"
    )
    print(
        f"GT / predicted boxes: "
        f"{detection['num_ground_truth_boxes']} / "
        f"{detection['num_predicted_boxes']}"
    )
    print(
        f"TP / FP / FN: "
        f"{detection['true_positives']} / "
        f"{detection['false_positives']} / "
        f"{detection['false_negatives']}"
    )
    print(
        f"Mean IoU:            "
        f"{detection['mean_iou_true_positive_matches']:.4f}"
    )
    print(
        f"Median IoU:          "
        f"{detection['median_iou_true_positive_matches']:.4f}"
    )
    print(
        f"Detection precision: "
        f"{detection['detection_precision']:.4f}"
    )
    print(
        f"Detection recall:    "
        f"{detection['detection_recall']:.4f}"
    )
    print(
        f"Detection F1:        "
        f"{detection['detection_f1']:.4f}"
    )

    print(
        f"Classification acc.: "
        f"{classification['accuracy']:.4f}"
    )
    print(
        f"Classification macro F1: "
        f"{classification['macro_f1']:.4f}"
    )
    print(
        f"Classification weighted F1: "
        f"{classification['weighted_f1']:.4f}"
    )

    print(
        "\nPer-class classification:"
    )

    for class_result in classification[
        "per_class"
    ]:
        print(
            f"  {class_result['class']}: "
            f"P={class_result['precision']:.4f}, "
            f"R={class_result['recall']:.4f}, "
            f"F1={class_result['f1_score']:.4f}, "
            f"support={class_result['support']}"
        )


def main() -> None:
    if not 0.0 <= IOU_THRESHOLD <= 1.0:
        raise ValueError(
            "IOU_THRESHOLD must be between 0 and 1."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Loading predictions: "
        f"{PREDICTIONS_JSONL}"
    )

    records = load_prediction_records(
        PREDICTIONS_JSONL,
        evaluate_top5
    )

    # ==================================================================
    # 1. GLOBAL EVALUATION — unchanged metrics across all datasets
    # ==================================================================
    overall_summary = save_evaluation_bundle(
        records=records,
        output_dir=OUTPUT_DIR,
    )

    print_summary(
        "OVERALL RESULTS — ALL DATASETS",
        overall_summary,
    )

    # ==================================================================
    # 2. SAME EVALUATION SEPARATELY FOR EACH SOURCE DATASET
    # ==================================================================
    datasets = sorted(
        {
            str(
                record["source_dataset"]
            )
            for record in records
        }
    )

    print(
        f"\nDatasets found: "
        f"{', '.join(datasets)}"
    )

    by_dataset_dir = (
        OUTPUT_DIR
        / "by_dataset"
    )
    by_dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_rows: list[
        dict[str, Any]
    ] = []

    for dataset_name in datasets:
        dataset_records = [
            record
            for record in records
            if str(
                record["source_dataset"]
            ) == dataset_name
        ]

        # Sanitize only for the directory name. The original source_dataset
        # string remains unchanged in all metric files.
        safe_dataset_name = "".join(
            character
            if character.isalnum()
            or character in ("-", "_")
            else "_"
            for character in dataset_name
        )

        dataset_output_dir = (
            by_dataset_dir
            / safe_dataset_name
        )

        dataset_summary = (
            save_evaluation_bundle(
                records=dataset_records,
                output_dir=dataset_output_dir,
            )
        )

        comparison_rows.append(
            make_dataset_comparison_row(
                dataset_name,
                dataset_summary,
            )
        )

        print_summary(
            f"DATASET: {dataset_name}",
            dataset_summary,
        )

    # ==================================================================
    # 3. ONE-FILE CROSS-DATASET COMPARISON
    # ==================================================================
    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        OUTPUT_DIR
        / "dataset_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # Also save the nested summaries in one JSON for convenient inspection.
    per_dataset_summary_json: dict[
        str,
        Any,
    ] = {}

    for dataset_name in datasets:
        safe_dataset_name = "".join(
            character
            if character.isalnum()
            or character in ("-", "_")
            else "_"
            for character in dataset_name
        )

        summary_path = (
            by_dataset_dir
            / safe_dataset_name
            / "evaluation_summary.json"
        )

        with summary_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            per_dataset_summary_json[
                dataset_name
            ] = json.load(
                file
            )

    per_dataset_summary_path = (
        OUTPUT_DIR
        / "evaluation_summary_by_dataset.json"
    )

    with per_dataset_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            per_dataset_summary_json,
            file,
            indent=2,
        )

    print(
        "\n[DONE]"
    )
    print(
        f"Overall evaluation: "
        f"{OUTPUT_DIR}"
    )
    print(
        f"Per-dataset evaluations: "
        f"{by_dataset_dir}"
    )
    print(
        f"Cross-dataset comparison: "
        f"{comparison_path}"
    )
    print(
        f"Per-dataset summary JSON: "
        f"{per_dataset_summary_path}"
    )


if __name__ == "__main__":
    main()
