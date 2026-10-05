#!/usr/bin/env python3
"""
Zero-shot prediction-conditioned temporal rollout with Qwen3-VL.

ARCHITECTURE
------------
The model predicts EVERY unique frame of each video sequentially.

For target frame t, Qwen always receives 16 image positions:
    positions 1-15 = temporal context
    position 16    = target frame

Warm-up:
    target frame 1:
        context = [F1] * 15
        target  = F1
        no context predictions exist yet

    target frame 2:
        context = [F1 + P1] + [F2] * 14
        target  = F2

    target frame 3:
        context = [F1 + P1, F2 + P2] + [F3] * 13
        target  = F3

    ...

    target frame 15:
        context = [F1 + P1, ..., F14 + P14] + [F15]
        target  = F15

    target frame 16:
        context = [F1 + P1, ..., F15 + P15]
        target  = F16

Normal rollout:
    target frame 17:
        context = [F2 + P2, ..., F16 + P16]
        target  = F17

where Pi is the MODEL'S prediction on Fi, never ground truth.

IMPORTANT
---------
Your JSON directory may contain overlapping 16-frame sliding-window JSON files.
This script DOES NOT infer independently from each JSON group. It first merges
those JSONs into one chronological timeline per video, then predicts each unique
frame exactly once so that rollout state is preserved across windows.

Ground truth is NEVER put into the model prompt. It is retained only in the saved
record so your existing evaluator can compare predictions against it.

Predicted bbox coordinates shown back to the model in context are the normalized
[0,1000] coordinates originally produced by Qwen for that frame.

GPU-only.

Suggested packages:
    pip install -U torch torchvision transformers accelerate qwen-vl-utils pillow tqdm
"""

import gc
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

JSON_DIR = Path("/home/marilyn/Downloads/data/JSON_temporal")

# Set to None if frame_path entries in the JSON already work on this machine.
FRAME_ROOT = Path("/home/marilyn/Downloads/data/frames")

OUTPUT_DIR = Path(
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/videoLevel_zeroShot_rollout_new_prompt/Qwen3VL-2B"
)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# "Train", "Val", "Test", or "All"
SPLIT = "Test"

MAX_NEW_TOKENS = 1024

# 0.0 => deterministic greedy decoding.
TEMPERATURE = 0.0

# Resume is safe because predictions are reloaded in chronological order and
# therefore reconstructed into rollout memory before later frames are inferred.
RESUME = True

# Optional quick tests:
LIMIT_VIDEOS = None          # e.g. 2
LIMIT_FRAMES_PER_VIDEO = None  # e.g. 25

# "auto", "flash_attention_2", "sdpa", or "eager"
ATTN_IMPLEMENTATION = "auto"

# Maximum number of people returned per target frame.
MAX_PEOPLE = 5


# ---------------- Visualization ----------------
# Save a copy of each TARGET frame with the MODEL predictions drawn on it.
# These annotated images are for inspection only and are NEVER fed back to Qwen.
SAVE_ANNOTATED_FRAMES = False

# Annotated images are stored per video under this directory.
ANNOTATED_FRAMES_DIR = OUTPUT_DIR / "annotated_frames"

# Drawing parameters.
ANNOTATION_LINE_WIDTH = 4
ANNOTATION_FONT_SIZE = 20


# ==============================================================================
# LABELS + PROMPT
# ==============================================================================

CLASS_NAMES = {
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
}


SYSTEM_PROMPT = """You are an expert in human-robot interaction, temporal human behavior analysis, human detection, and visual grounding.

Your task is TEMPORAL HUMAN DETECTION AND INTERACTION-READINESS CLASSIFICATION.

You will receive exactly 16 ordered image positions.

============================================================
TEMPORAL STRUCTURE
==================

* Images 1 through 15 are CONTEXT.
* Image 16, the LAST image, is the TARGET frame.
* You must output predictions ONLY for image 16.

At the beginning of a video, fewer than 15 real previous frames may exist.
In that case, some context positions contain repeated copies of the TARGET image.

These repeated images are PADDING ONLY:

* they are NOT previous observations,
* they contain no previous predictions,
* they must NOT be interpreted as temporal evidence.

For real previous context frames, you will receive predictions produced by this same model during earlier rollout steps.

These previous annotations are MODEL PREDICTIONS, NOT ground truth.
They may contain errors.

Use previous frames and predictions as temporal evidence, but ALWAYS verify the person, bounding box, and state directly from the TARGET image.

Never copy a previous bounding box merely because the same person was present earlier.

============================================================
CONTEXT PREDICTIONS
===================

For a real previous context frame, metadata will contain:

* person_ID
* bbox
* label

Context bbox format:

[x1, y1, x2, y2]

Context labels are:

* not_interaction_ready
* interaction_ready
* interaction_ongoing

Use temporal information to reason about:

* whether a person was previously present,
* whether the person is approaching or moving away,
* changes in body orientation,
* gaze or head direction when visible,
* stopping or waiting,
* gestures,
* transitions between interaction-readiness states,
* whether an interaction appears to have started or ended.

Temporal information should help determine person identity continuity over time and interaction-readiness state.

Use temporal context to help identify and interpret people, but determine each TARGET-frame bounding box from the person's visible location in image 16.

A previously visible person may have moved.
A previously visible person may have disappeared.
A new person may have appeared.
A previous bounding box may no longer be accurate.

============================================================
TARGET FRAME — IMAGE 16
=======================

Perform the following steps ONLY for image 16:

1. Inspect image 16 directly.

2. Identify the humans visibly present in image 16.

3. Select at most 5 humans relevant to the robot.
   If more than 5 humans are visible, prioritize people who appear closest to the robot or most directly involved in the interaction.

4. For EACH selected person:

   * localize that person directly in image 16,
   * predict one tight bounding box around the person's FULL VISIBLE EXTENT,
   * assign one interaction-readiness label.

Do not predict a person only because they appeared in a previous frame.
The person must be visible in image 16.

============================================================
BOUNDING BOXES — VERY IMPORTANT
===============================

Every target bounding box MUST use:

[x1, y1, x2, y2]

where:

* x1 = left edge of the person
* y1 = top edge of the person
* x2 = right edge of the person
* y2 = bottom edge of the person

Coordinates MUST be normalized to the range [0,1000] relative to IMAGE 16 itself.

Coordinate system:

* left edge of image = x = 0
* right edge of image = x = 1000
* top edge of image = y = 0
* bottom edge of image = y = 1000

Therefore:

* x1 < x2
* y1 < y2
* 0 <= x1,x2,y1,y2 <= 1000

IMPORTANT:
- Predict ONLY image position 16.
- Do NOT output predictions for context positions 1-15.
- Re-localize the person directly in image 16.
- Do NOT use coordinates from a context image.
- Do NOT copy a previous bounding box.
- Padding copies of the target frame are not previous observations.
- Previous context annotations are predictions, not ground truth.
- Detect the target people from the target image itself.
- Do not blindly copy previous boxes or labels.
- Return each target-frame person exactly once.
- Return no more than 5 people.

============================================================
INTERACTION-READINESS LABELS
============================

Choose exactly ONE of the following labels for each selected person.

not_interaction_ready:
The person is not currently signaling readiness or intention to initiate an interaction with the robot.

interaction_ready:
The person appears ready or intends to initiate an interaction with the robot, but the interaction itself has not yet started.

interaction_ongoing:
The person is already actively engaged in an interaction with the robot.

Use both the current image and temporal context when determining the label.

The target image is the primary evidence.
Previous predictions are supporting evidence only.

============================================================
PERSON IDs AND TEMPORAL IDENTITY
==========

person_ID represents the same physical person across frames within the same video.

Use previous context frames and previous predictions to maintain identity continuity over time.

When a person visible in the TARGET frame corresponds to a person seen in previous real context frames, reuse that person's previous person_ID.

Use the following cues for temporal correspondence:
- visual appearance,
- spatial location,
- motion trajectory,
- body shape,
- clothing,
- relative position to other people,
- previous bounding boxes and labels.

If a new person appears in the TARGET frame who was not present in the previous real context frames, assign the smallest unused positive integer person_ID.

Do not change a person's ID simply because their position, pose, bounding box, or interaction-readiness label changes.

Do not reuse the ID of a person who is still visible for another person.

If identity correspondence is uncertain, choose the assignment that is most temporally consistent with the previous frames.

The bounding box itself must still be localized from the person's visible position in the TARGET frame. Tracking determines WHO the person is; target-frame grounding determines WHERE they are now.

============================================================
FINAL CHECK BEFORE ANSWERING
============================

Before producing the answer, verify:

1. I am predicting ONLY image 16.
2. Every returned person is actually visible in image 16.
3. Every person appears exactly once.
4. I returned no more than 5 people.
5. Every bbox is localized from image 16 itself.
6. Every bbox uses [x1,y1,x2,y2].
7. Every bbox coordinate is between 0 and 1000.
8. Every bbox covers the full visible extent of the person.
9. Every label is one of the three allowed labels.
10. My response contains JSON only.

============================================================
OUTPUT FORMAT
=============

Return ONLY valid JSON.

Do NOT include:

* markdown,
* code fences,
* explanations,
* reasoning,
* comments,
* additional text.

Required schema:

{"people":[
{
"person_ID":1,
"bbox":[x1,y1,x2,y2],
"label":"interaction_ready"
},
{
"person_ID":2,
"bbox":[x1,y1,x2,y2],
"label":"not_interaction_ready"
},
{
"person_ID":3,
"bbox":[x1,y1,x2,y2],
"label":"interaction_ongoing"
}
]}

If no human is visible in image 16, return exactly:

{"people":[]}
"""


# ==============================================================================
# MODEL
# ==============================================================================

def check_cuda() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This script is intentionally GPU-only."
        )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[GPU] CUDA visible to PyTorch: {torch.version.cuda}")
    print(f"[GPU] dtype: {dtype}")

    return dtype


def load_model(dtype: torch.dtype):
    print(f"[MODEL] Loading {MODEL_NAME}")

    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }

    if ATTN_IMPLEMENTATION != "auto":
        kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        **kwargs,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(f"[MODEL] loaded on {model.device}")
    return model, processor


# ==============================================================================
# PATH HELPERS
# ==============================================================================

def dataset_name_from_json_file(
    json_path: Path,
    video_name: str,
) -> str:
    # e.g. AVIDAR__AVIDAR_004__group_000001.json -> AVIDAR
    if "__" in json_path.stem:
        return json_path.stem.split("__", 1)[0]

    if "_" in video_name:
        return video_name.split("_", 1)[0]

    return video_name


def windows_basename(path_string: str) -> str:
    return re.split(r"[\\/]", path_string)[-1]


def resolve_frame_path(
    original_path: str,
    frame_root: Optional[Path],
    dataset_name: str,
    video_name: str,
) -> Path:
    direct = Path(original_path)

    if direct.exists():
        return direct.resolve()

    if frame_root is None:
        raise FileNotFoundError(
            f"Frame path does not exist: {original_path}\n"
            "Set FRAME_ROOT if the JSONs were produced on another machine."
        )

    filename = windows_basename(original_path)

    candidates = [
        frame_root / dataset_name / video_name / filename,
        frame_root / video_name / filename,
        frame_root / dataset_name / filename,
        frame_root / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve frame.\n"
        f"Original JSON path: {original_path}\n"
        f"FRAME_ROOT: {frame_root}\n"
        "Tried:\n  - "
        + "\n  - ".join(str(path) for path in candidates)
    )


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


# ==============================================================================
# BOUNDING BOX HELPERS
# ==============================================================================

def denormalize_box_1000(
    box: List[float],
    width: int,
    height: int,
) -> List[float]:
    x1, y1, x2, y2 = map(float, box)

    x1 = max(0.0, min(x1, 1000.0))
    x2 = max(0.0, min(x2, 1000.0))
    y1 = max(0.0, min(y1, 1000.0))
    y2 = max(0.0, min(y2, 1000.0))

    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    return [
        left * width / 1000.0,
        top * height / 1000.0,
        right * width / 1000.0,
        bottom * height / 1000.0,
    ]


# ==============================================================================
# LABEL / JSON PARSING
# ==============================================================================

def normalize_predicted_label(value: Any) -> str:
    label = str(value).strip().lower()

    if label == "interaction_done":
        label = "not_interaction_ready"

    if label not in CLASS_NAMES:
        raise ValueError(
            f"Unsupported predicted label: {value!r}"
        )

    return label


def normalize_gt_label(value: Any) -> str:
    label = str(value).strip().lower()

    if label == "interaction_done":
        return "not_interaction_ready"

    return label


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        parsed = json.loads(candidate)

        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model response is not valid JSON.")


def parse_predicted_people(
    parsed: Dict[str, Any],
    target_width: int,
    target_height: int,
) -> List[Dict[str, Any]]:
    raw_people = parsed.get("people")

    if not isinstance(raw_people, list):
        raise ValueError(
            "Model JSON must contain a list field named 'people'."
        )

    predictions: List[Dict[str, Any]] = []

    for index, person in enumerate(raw_people[:MAX_PEOPLE]):
        if not isinstance(person, dict):
            raise ValueError(
                f"people[{index}] is not a JSON object."
            )

        if "bbox" not in person:
            raise ValueError(
                f"people[{index}] is missing bbox."
            )

        if "label" not in person:
            raise ValueError(
                f"people[{index}] is missing label."
            )

        bbox = person["bbox"]

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            raise ValueError(
                f"people[{index}].bbox must have four coordinates."
            )

        normalized_bbox = [float(v) for v in bbox]

        pixel_bbox = denormalize_box_1000(
            normalized_bbox,
            target_width,
            target_height,
        )

        x1, y1, x2, y2 = pixel_bbox
        if x2 <= x1 or y2 <= y1:
            continue

        # We deliberately standardize IDs to sequential frame-local indices.
        person_id = len(predictions) + 1

        predictions.append(
            {
                "person_id": person_id,
                "bbox_2d": pixel_bbox,
                "label": normalize_predicted_label(person["label"]),
                "bbox_normalized_1000": normalized_bbox,
            }
        )

    return predictions


def build_ground_truth_people(
    frame: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ground_truth: List[Dict[str, Any]] = []

    for person in frame.get("people", []):
        original_label = str(
            person["label"]
        ).strip()

        ground_truth.append(
            {
                "person_id": int(person["person_ID"]),
                "bbox_xyxy": [
                    float(value)
                    for value in person["bbox"]
                ],
                "label_original": original_label,
                "label_3class": normalize_gt_label(
                    original_label
                ),
            }
        )

    return ground_truth


# ==============================================================================
# DATASET RECONSTRUCTION
# ==============================================================================

def split_matches(
    group_split: str,
    requested_split: str,
) -> bool:
    if requested_split.lower() == "all":
        return True

    return (
        group_split.strip().lower()
        == requested_split.strip().lower()
    )


def load_video_timelines() -> Dict[str, Dict[str, Any]]:
    """
    Merge all overlapping 16-frame JSON groups into one unique chronological
    timeline per source video.

    Returns:
        {
          unique_video_key: {
              "video_name": ...,
              "source_dataset": ...,
              "split": ...,
              "frames": [unique frames sorted chronologically]
          }
        }

    Each frame keeps GT in memory ONLY for evaluator output. build_messages()
    never receives GT annotations.
    """

    json_files = sorted(JSON_DIR.rglob("*.json"))

    videos: Dict[str, Dict[str, Any]] = {}

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as file:
                group = json.load(file)
        except Exception as error:
            print(f"[SKIP] {json_path.name}: {error}")
            continue

        required = {
            "video_name",
            "frame_group",
            "split",
            "frames",
        }

        if not required.issubset(group.keys()):
            continue

        if not split_matches(str(group["split"]), SPLIT):
            continue

        video_name = str(group["video_name"])
        dataset_name = dataset_name_from_json_file(
            json_path,
            video_name,
        )

        # Avoid collisions if two datasets use the same video_name.
        video_key = f"{dataset_name}::{video_name}"

        if video_key not in videos:
            videos[video_key] = {
                "video_name": video_name,
                "source_dataset": dataset_name,
                "split": str(group["split"]),
                "frames_by_number": {},
            }

        video_entry = videos[video_key]

        if (
            video_entry["split"].strip().lower()
            != str(group["split"]).strip().lower()
        ):
            raise ValueError(
                f"Inconsistent split for video {video_key}: "
                f"{video_entry['split']} vs {group['split']}"
            )

        for frame in group["frames"]:
            frame_number = int(frame["frame_number"])

            # Resolve path now, while the originating JSON path is available.
            resolved_path = resolve_frame_path(
                frame["frame_path"],
                FRAME_ROOT,
                dataset_name,
                video_name,
            )

            frame_copy = dict(frame)
            frame_copy["_resolved_path"] = str(resolved_path)
            frame_copy["_source_json"] = json_path.name

            existing = video_entry[
                "frames_by_number"
            ].get(frame_number)

            if existing is None:
                video_entry["frames_by_number"][
                    frame_number
                ] = frame_copy
            else:
                # Overlapping windows should describe the same physical frame.
                # Keep the first copy but sanity-check the path basename.
                old_name = windows_basename(
                    existing["frame_path"]
                )
                new_name = windows_basename(
                    frame_copy["frame_path"]
                )

                if old_name != new_name:
                    print(
                        f"[WARNING] {video_key} frame {frame_number} "
                        f"has differing paths: {old_name} vs {new_name}"
                    )

    # Convert dicts into sorted frame arrays.
    final_videos: Dict[str, Dict[str, Any]] = {}

    for video_key, entry in videos.items():
        frames = [
            entry["frames_by_number"][frame_number]
            for frame_number in sorted(
                entry["frames_by_number"].keys()
            )
        ]

        final_videos[video_key] = {
            "video_name": entry["video_name"],
            "source_dataset": entry["source_dataset"],
            "split": entry["split"],
            "frames": frames,
        }

    # Stable ordering.
    final_videos = dict(
        sorted(final_videos.items(), key=lambda x: x[0])
    )

    if LIMIT_VIDEOS is not None:
        final_videos = dict(
            list(final_videos.items())[:LIMIT_VIDEOS]
        )

    print(f"[DATA] JSON files found: {len(json_files)}")
    print(
        f"[DATA] Reconstructed {len(final_videos)} videos "
        f"for split={SPLIT}"
    )

    for key, video in final_videos.items():
        print(
            f"       {key}: {len(video['frames'])} unique frames"
        )

    return final_videos


# ==============================================================================
# ROLLOUT CONTEXT
# ==============================================================================

def context_prediction_for_prompt(
    prediction_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a previous saved model prediction into the compact form supplied
    back to Qwen as context.

    Crucially, this uses predicted_people only. Ground truth is never read here.
    """
    people = []

    for person in prediction_record.get(
        "predicted_people",
        [],
    ):
        bbox = person.get("bbox_normalized_1000")

        if bbox is None:
            # Old/incomplete record: cannot safely inject a pixel-space bbox
            # because context frames may have different dimensions.
            continue

        people.append(
            {
                "person_ID": person.get("person_id"),
                "bbox": [
                    round(float(value), 2)
                    for value in bbox
                ],
                "label": person.get("label"),
            }
        )

    return {
        "people": people,
    }


def build_rollout_context(
    frames: List[Dict[str, Any]],
    target_index: int,
    rollout_memory: Dict[int, Dict[str, Any]],
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """
    Build exactly 15 context image positions.

    target_index is zero-based.

    For target_index < 15:
        [all real previous frames] + [target repeated until length=15]

    For target_index >= 15:
        [previous 15 real frames]

    Returns:
        context_paths
        context_metadata

    context_metadata says which positions are genuine previous frames and which
    positions are padding copies of the current target.
    """

    target_frame = frames[target_index]
    target_path = Path(target_frame["_resolved_path"])

    history_start = max(0, target_index - 15)
    history_indices = list(
        range(history_start, target_index)
    )

    context_paths: List[Path] = []
    context_metadata: List[Dict[str, Any]] = []

    # Real previous frames, all of which must already have a rollout prediction.
    for history_index in history_indices:
        history_frame = frames[history_index]
        history_number = int(
            history_frame["frame_number"]
        )

        if history_number not in rollout_memory:
            raise RuntimeError(
                f"Missing rollout prediction for previous frame "
                f"{history_number} while predicting "
                f"{target_frame['frame_number']}."
            )

        previous_record = rollout_memory[
            history_number
        ]

        context_paths.append(
            Path(history_frame["_resolved_path"])
        )

        context_metadata.append(
            {
                "kind": "previous_frame",
                "dataset_frame_number": history_number,
                "prediction": context_prediction_for_prompt(
                    previous_record
                ),
            }
        )

    # Warm-up padding: repeat current target image without annotations.
    while len(context_paths) < 15:
        context_paths.append(target_path)

        context_metadata.append(
            {
                "kind": "target_padding",
                "dataset_frame_number": int(
                    target_frame["frame_number"]
                ),
                "prediction": None,
            }
        )

    assert len(context_paths) == 15
    assert len(context_metadata) == 15

    return context_paths, context_metadata


def build_context_text(
    video: Dict[str, Any],
    target_frame: Dict[str, Any],
    context_metadata: List[Dict[str, Any]],
) -> str:
    """
    Textually align each context position with either:
      - a previous model prediction, or
      - an explicitly unannotated target-frame padding copy.
    """

    lines: List[str] = []

    lines.append(f"Video: {video['video_name']}")
    lines.append(
        f"Target dataset frame number: {int(target_frame['frame_number'])}"
    )
    lines.append("")
    lines.append(
        "The 16 supplied image positions are ordered exactly as listed below."
    )
    lines.append(
        "Positions 1-15 are context; position 16 is the target."
    )
    lines.append("")
    lines.append("CONTEXT POSITION METADATA:")

    for position, metadata in enumerate(
        context_metadata,
        start=1,
    ):
        if metadata["kind"] == "target_padding":
            lines.append(
                f"Position {position}: TARGET-FRAME PADDING COPY "
                f"(dataset frame {metadata['dataset_frame_number']}). "
                "This is NOT a previous observation and has NO prediction."
            )
            continue

        prediction_json = json.dumps(
            metadata["prediction"],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        lines.append(
            f"Position {position}: previous dataset frame "
            f"{metadata['dataset_frame_number']}; "
            f"MODEL PREDICTION = {prediction_json}"
        )

    lines.append("")
    lines.append(
        "Position 16: TARGET frame "
        f"(dataset frame {int(target_frame['frame_number'])})."
    )
    lines.append(
        "No target annotation is supplied. Detect and classify the target people."
    )
    lines.append(
        "Return target bounding boxes in normalized [0,1000] xyxy coordinates."
    )

    return "\n".join(lines)


def build_messages(
    video: Dict[str, Any],
    target_frame: Dict[str, Any],
    context_paths: List[Path],
    context_metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    target_path = Path(target_frame["_resolved_path"])

    # Exactly 15 context positions + 1 target position.
    model_paths = context_paths + [target_path]

    context_text = build_context_text(
        video,
        target_frame,
        context_metadata,
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [
                        str(path.resolve())
                        for path in model_paths
                    ],
                },
                {
                    "type": "text",
                    "text": context_text,
                },
            ],
        },
    ]


# ==============================================================================
# MODEL INFERENCE
# ==============================================================================

@torch.inference_mode()
def infer_target(
    model,
    processor,
    video: Dict[str, Any],
    target_frame: Dict[str, Any],
    context_paths: List[Path],
    context_metadata: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    str,
    Optional[str],
    int,
    int,
]:
    target_path = Path(target_frame["_resolved_path"])
    target_width, target_height = get_image_size(
        target_path
    )

    messages = build_messages(
        video,
        target_frame,
        context_paths,
        context_metadata,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    (
        image_inputs,
        video_inputs,
        video_kwargs,
    ) = process_vision_info(
        messages,
        image_patch_size=(
            processor.image_processor.patch_size
        ),
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if video_inputs is not None:
        videos, video_metadatas = zip(*video_inputs)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        videos = None
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=videos,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )

    inputs = inputs.to(model.device)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    if TEMPERATURE > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_p": 0.9,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    generated_ids = model.generate(
        **inputs,
        **generation_kwargs,
    )

    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(inputs.input_ids, generated_ids)
    ]

    raw_response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parse_error = None

    try:
        parsed = extract_json_object(
            raw_response
        )

        predicted_people = parse_predicted_people(
            parsed,
            target_width,
            target_height,
        )

    except Exception as error:
        predicted_people = []
        parse_error = (
            f"{type(error).__name__}: {error}"
        )

    return (
        predicted_people,
        raw_response,
        parse_error,
        target_width,
        target_height,
    )


# ==============================================================================
# PREDICTION VISUALIZATION
# ==============================================================================

def get_annotation_font():
    """Return a readable font, falling back to Pillow's default font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, ANNOTATION_FONT_SIZE)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()


def annotated_frame_output_path(
    video: Dict[str, Any],
    frame_number: int,
) -> Path:
    """Return the path for an annotated prediction frame."""
    video_dir = (
        ANNOTATED_FRAMES_DIR
        / safe_video_filename(
            video["source_dataset"],
            video["video_name"],
        )
    )

    video_dir.mkdir(parents=True, exist_ok=True)

    return (
        video_dir
        / f"frame_{frame_number:08d}__annotated.jpg"
    )


def save_annotated_prediction_frame(
    video: Dict[str, Any],
    target_frame: Dict[str, Any],
    predicted_people: List[Dict[str, Any]],
) -> Optional[Path]:
    """
    Save the raw target frame with MODEL predictions drawn on it.

    The visualization is produced only after inference and is never used as
    model input or rollout context.
    """
    if not SAVE_ANNOTATED_FRAMES:
        return None

    target_path = Path(target_frame["_resolved_path"])
    frame_number = int(target_frame["frame_number"])
    output_path = annotated_frame_output_path(video, frame_number)

    with Image.open(target_path) as source:
        image = source.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = get_annotation_font()

    for prediction in predicted_people:
        bbox = prediction.get("bbox_2d")

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(value) for value in bbox]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=ANNOTATION_LINE_WIDTH,
        )

        person_id = prediction.get("person_id", "?")
        label = prediction.get("label", "unknown")
        caption = f"ID {person_id}: {label}"

        try:
            text_box = draw.textbbox((x1, y1), caption, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
        except AttributeError:
            text_width, text_height = draw.textsize(caption, font=font)

        padding = 4
        text_y = y1 - text_height - 2 * padding
        if text_y < 0:
            text_y = y1

        text_x = max(
            0,
            min(
                x1,
                image.width - text_width - 2 * padding,
            ),
        )

        background_box = [
            text_x,
            text_y,
            text_x + text_width + 2 * padding,
            text_y + text_height + 2 * padding,
        ]

        draw.rectangle(background_box, fill="red")
        draw.text(
            (text_x + padding, text_y + padding),
            caption,
            fill="white",
            font=font,
        )

    image.save(output_path, quality=95)
    return output_path


# ==============================================================================
# OUTPUT
# ==============================================================================

def safe_video_filename(
    source_dataset: str,
    video_name: str,
) -> str:
    value = f"{source_dataset}__{video_name}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def frame_output_path(
    video: Dict[str, Any],
    frame_number: int,
) -> Path:
    video_dir = (
        OUTPUT_DIR
        / safe_video_filename(
            video["source_dataset"],
            video["video_name"],
        )
    )

    video_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        video_dir
        / f"frame_{frame_number:08d}__predictions.json"
    )


def save_record(
    path: Path,
    record: Dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_existing_record(
    path: Path,
) -> Dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_record(
    video: Dict[str, Any],
    target_frame: Dict[str, Any],
    predicted_people: List[Dict[str, Any]],
    raw_response: str,
    parse_error: Optional[str],
    target_width: int,
    target_height: int,
    context_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    previous_real_frames = [
        metadata["dataset_frame_number"]
        for metadata in context_metadata
        if metadata["kind"] == "previous_frame"
    ]

    padding_count = sum(
        metadata["kind"] == "target_padding"
        for metadata in context_metadata
    )

    return {
        # Existing evaluator-compatible fields:
        "source_dataset": video["source_dataset"],
        "video_id": str(video["video_name"]),
        "frame_id": int(target_frame["frame_number"]),
        "ground_truth_people": build_ground_truth_people(
            target_frame
        ),
        "predicted_people": predicted_people,
        "image_error": None,
        "parse_error": parse_error,

        # Experiment metadata:
        "split": str(video["split"]),
        "model": MODEL_NAME,
        "zero_shot": True,
        "temporal_method": "prediction_conditioned_rollout",
        "target_frame_number": int(
            target_frame["frame_number"]
        ),
        "target_image_width": target_width,
        "target_image_height": target_height,

        # Rollout metadata:
        "rollout_previous_frame_numbers": previous_real_frames,
        "rollout_history_length": len(previous_real_frames),
        "rollout_padding_count": padding_count,
        "rollout_context": context_metadata,

        # Raw response for debugging:
        "raw_model_response": raw_response,
    }


def build_error_record(
    video: Dict[str, Any],
    target_frame: Dict[str, Any],
    error: Exception,
    context_metadata: List[Dict[str, Any]],
) -> Dict[str, Any]:
    previous_real_frames = [
        metadata["dataset_frame_number"]
        for metadata in context_metadata
        if metadata["kind"] == "previous_frame"
    ]

    padding_count = sum(
        metadata["kind"] == "target_padding"
        for metadata in context_metadata
    )

    return {
        "source_dataset": video["source_dataset"],
        "video_id": str(video["video_name"]),
        "frame_id": int(target_frame["frame_number"]),
        "ground_truth_people": build_ground_truth_people(
            target_frame
        ),
        "predicted_people": [],
        "image_error": (
            f"{type(error).__name__}: {error}"
        ),
        "parse_error": None,
        "split": str(video["split"]),
        "model": MODEL_NAME,
        "zero_shot": True,
        "temporal_method": "prediction_conditioned_rollout",
        "target_frame_number": int(
            target_frame["frame_number"]
        ),
        "rollout_previous_frame_numbers": previous_real_frames,
        "rollout_history_length": len(previous_real_frames),
        "rollout_padding_count": padding_count,
        "rollout_context": context_metadata,
        "raw_model_response": None,
    }


def rebuild_combined_jsonl() -> Path:
    combined_path = OUTPUT_DIR / "predictions.jsonl"

    prediction_files = sorted(
        path
        for path in OUTPUT_DIR.rglob(
            "frame_*__predictions.json"
        )
    )

    with combined_path.open(
        "w",
        encoding="utf-8",
    ) as combined:
        for path in prediction_files:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                record = json.load(file)

            combined.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return combined_path


# ==============================================================================
# VIDEO ROLLOUT
# ==============================================================================

def run_video(
    model,
    processor,
    video_key: str,
    video: Dict[str, Any],
) -> None:
    frames = video["frames"]

    if LIMIT_FRAMES_PER_VIDEO is not None:
        frames = frames[:LIMIT_FRAMES_PER_VIDEO]

    # Keyed by actual dataset frame number.
    rollout_memory: Dict[
        int,
        Dict[str, Any]
    ] = {}

    print(
        f"\n[VIDEO] {video_key} "
        f"({len(frames)} frames)"
    )

    for target_index in tqdm(
        range(len(frames)),
        desc=video["video_name"],
        leave=False,
    ):
        target_frame = frames[target_index]
        frame_number = int(
            target_frame["frame_number"]
        )

        output_path = frame_output_path(
            video,
            frame_number,
        )

        # IMPORTANT:
        # Even when resuming, load the old prediction into rollout_memory.
        # Otherwise subsequent frames would lose their previous prediction.
        if RESUME and output_path.exists():
            try:
                record = load_existing_record(
                    output_path
                )

                rollout_memory[
                    frame_number
                ] = record

                # If the prediction already exists but its visualization does not,
                # recreate the annotated frame without rerunning Qwen.
                if SAVE_ANNOTATED_FRAMES:
                    annotated_path = annotated_frame_output_path(
                        video,
                        frame_number,
                    )

                    if not annotated_path.exists():
                        try:
                            save_annotated_prediction_frame(
                                video=video,
                                target_frame=target_frame,
                                predicted_people=record.get(
                                    "predicted_people",
                                    [],
                                ),
                            )
                        except Exception as annotation_error:
                            print(
                                f"\n[ANNOTATION WARNING] "
                                f"{video_key} / frame {frame_number}: "
                                f"{annotation_error}"
                            )

                continue

            except Exception as error:
                print(
                    f"\n[RESUME WARNING] Could not load "
                    f"{output_path}: {error}. Recomputing."
                )

        context_paths, context_metadata = (
            build_rollout_context(
                frames,
                target_index,
                rollout_memory,
            )
        )

        try:
            (
                predicted_people,
                raw_response,
                parse_error,
                target_width,
                target_height,
            ) = infer_target(
                model=model,
                processor=processor,
                video=video,
                target_frame=target_frame,
                context_paths=context_paths,
                context_metadata=context_metadata,
            )

            record = build_record(
                video=video,
                target_frame=target_frame,
                predicted_people=predicted_people,
                raw_response=raw_response,
                parse_error=parse_error,
                target_width=target_width,
                target_height=target_height,
                context_metadata=context_metadata,
            )

        except Exception as error:
            record = build_error_record(
                video=video,
                target_frame=target_frame,
                error=error,
                context_metadata=context_metadata,
            )

            print(
                f"\n[ERROR] {video_key} / "
                f"frame {frame_number}: {error}"
            )

        # Save immediately so a crash loses at most the current frame.
        save_record(
            output_path,
            record,
        )

        # Optional visualization of the MODEL prediction on the raw target frame.
        # This output is never used as rollout input.
        if SAVE_ANNOTATED_FRAMES:
            try:
                save_annotated_prediction_frame(
                    video=video,
                    target_frame=target_frame,
                    predicted_people=record.get(
                        "predicted_people",
                        [],
                    ),
                )
            except Exception as annotation_error:
                print(
                    f"\n[ANNOTATION WARNING] "
                    f"{video_key} / frame {frame_number}: "
                    f"{annotation_error}"
                )

        # This is the key rollout operation:
        # the current prediction becomes context memory for future frames.
        rollout_memory[
            frame_number
        ] = record

        if record.get("parse_error"):
            print(
                f"\n[PARSE ERROR] {video_key} / "
                f"frame {frame_number}: "
                f"{record['parse_error']}"
            )

        gc.collect()
        torch.cuda.empty_cache()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dtype = check_cuda()
    model, processor = load_model(dtype)

    videos = load_video_timelines()

    for video_key, video in videos.items():
        run_video(
            model=model,
            processor=processor,
            video_key=video_key,
            video=video,
        )

    combined_path = rebuild_combined_jsonl()

    print("\n[DONE]")
    print(
        f"Per-frame rollout outputs: {OUTPUT_DIR}"
    )
    print(
        f"Combined evaluator JSONL: {combined_path}"
    )

    if SAVE_ANNOTATED_FRAMES:
        print(
            f"Annotated prediction frames: {ANNOTATED_FRAMES_DIR}"
        )


if __name__ == "__main__":
    main()
