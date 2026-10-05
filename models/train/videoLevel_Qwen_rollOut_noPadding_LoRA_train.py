#!/usr/bin/env python3
"""
LoRA fine-tuning for Qwen3-VL on prediction-conditioned temporal interaction-readiness.

This training script matches the accompanying rollout zero-shot inference task:

    positions 1-15 = temporal context
    position 16    = target frame
    output          = JSON for target frame only

TRAINING VS. INFERENCE
----------------------
Inference feeds the model's OWN earlier predictions back as context.

Training uses TEACHER FORCING only where rollout inference would already have
a model prediction available:
    - prediction begins at frame 16, using frames 1-15 as unannotated context,
    - for later targets, only context frames from frame 16 onward receive
      teacher-forced ground-truth annotations formatted as prior model predictions,
    - pre-rollout context frames remain unannotated, exactly as at inference,
    - the loss is computed ONLY on the assistant target-frame JSON answer.

This is intentional. Generating previous predictions inside the same gradient
step would make training autoregressive across an entire video and would be
non-differentiable through generated discrete outputs. Teacher forcing is the
standard practical analogue.

IMPORTANT:
The model never receives the TARGET frame's ground truth in its prompt.
Target ground truth appears only as the supervised assistant answer.

The script first merges overlapping 16-frame JSON windows into one chronological
timeline per source video, exactly as the rollout inference script does.

Suggested packages:
    pip install -U torch torchvision transformers accelerate peft qwen-vl-utils pillow tqdm

For Qwen3-VL, use a recent transformers build compatible with your inference
environment.
"""

import gc
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)


# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

JSON_DIR = Path("/home/marilyn/Downloads/data/JSON_temporal")

# Set to None if frame_path entries in the JSON already work on this machine.
FRAME_ROOT = Path("/home/marilyn/Downloads/data/frames")

OUTPUT_DIR = Path(
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/videoLevel_LoRA_rollout_noPadding/Qwen3VL_2B_Instruct_R8"
)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# Training data only.
SPLIT = "Train"
VALIDATION_SPLIT = "Val"

# ---------------- LoRA ----------------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Match the previous trainer: adapt attention projections only.
# The vision encoder and MLP projections remain frozen.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# ---------------- Optimization ----------------
EPOCHS = 1
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.01

# Keep micro-batch = 1 because each example contains 16 visual positions.
GRADIENT_ACCUMULATION_STEPS = 8

WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0

# Save adapter checkpoints every N optimizer updates.
SAVE_EVERY_OPTIMIZER_STEPS = 200

# Log every N optimizer updates.
LOG_EVERY_OPTIMIZER_STEPS = 10

# Match the previous trainer's checkpoint retention behavior.
SAVE_TOTAL_LIMIT = 3

# Lightweight validation cadence. A 32-example fixed subset every 1000
# optimizer steps adds only a small runtime overhead.
EVAL_EVERY_OPTIMIZER_STEPS = 1000
EVAL_MAX_SAMPLES = 32

# Resume from the latest checkpoint under OUTPUT_DIR when available.
RESUME_FROM_LAST_CHECKPOINT = True

DATALOADER_NUM_WORKERS = 0

# Optional processor pixel limits, matching the previous trainer.
MAX_PIXELS = 448 * 448
MIN_PIXELS = None

SEED = 42

# "auto", "flash_attention_2", "sdpa", or "eager"
ATTN_IMPLEMENTATION = "auto"

# Strongly recommended for video LoRA.
GRADIENT_CHECKPOINTING = False

# Maximum number of people in the supervised target JSON.
MAX_PEOPLE = 5

# Optional quick tests.
LIMIT_VIDEOS = None          # e.g. 2
LIMIT_TRAIN_SAMPLES = None   # e.g. 20

# If True, randomize the order of target frames between epochs.
# Teacher-forced context is constructed from GT, so samples do not have to be
# processed chronologically during training.
SHUFFLE_EACH_EPOCH = True


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

Rollout prediction begins only at frame 16 of each video.

Therefore:

* the first predicted target (frame 16) receives frames 1-15 as real visual context,
  but NONE of those context frames has a previous model prediction;
* for frame 17, only frame 16 has a previous model prediction;
* for frame 18, frames 16 and 17 have previous model predictions;
* this continues until all 15 context positions correspond to frames that were
  previously predicted by the model.

A context frame without a MODEL PREDICTION is still a real previous observation
and should be used as visual temporal evidence. It simply has no prior model
annotation attached to it.

When a previous prediction is supplied, it was produced by this same model during
an earlier rollout step. These previous annotations are MODEL PREDICTIONS, NOT
ground truth, and they may contain errors.

Use previous frames and available predictions as temporal evidence, but ALWAYS
verify the person, bounding box, and state directly from the TARGET image.

Never copy a previous bounding box merely because the same person was present earlier.

============================================================
CONTEXT PREDICTIONS
===================

For a previous context frame that already has a rollout prediction, metadata
will contain:

* person_ID
* bbox
* label

Earlier pre-rollout context frames may explicitly have NO MODEL PREDICTION.

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
- Context frames before rollout begins may legitimately have no prior prediction.
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
# REPRODUCIBILITY / CUDA
# ==============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# ==============================================================================
# PATH HELPERS
# ==============================================================================

def dataset_name_from_json_file(
    json_path: Path,
    video_name: str,
) -> str:
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


_IMAGE_SIZE_CACHE: Dict[str, Tuple[int, int]] = {}


def get_image_size(path: Path) -> Tuple[int, int]:
    key = str(path)

    if key not in _IMAGE_SIZE_CACHE:
        with Image.open(path) as image:
            _IMAGE_SIZE_CACHE[key] = image.size

    return _IMAGE_SIZE_CACHE[key]


# ==============================================================================
# LABEL / BOX HELPERS
# ==============================================================================

def normalize_label(value: Any) -> str:
    label = str(value).strip().lower()

    aliases = {
        "not_ready": "not_interaction_ready",
        "ready": "interaction_ready",
        "ongoing": "interaction_ongoing",
        "interaction_done": "not_interaction_ready",
    }
    label = aliases.get(label, label)

    if label not in CLASS_NAMES:
        raise ValueError(f"Unsupported label: {value!r}")

    return label


def normalize_pixel_box_to_1000(
    box: List[float],
    width: int,
    height: int,
) -> List[float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")

    x1, y1, x2, y2 = map(float, box)

    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    left = max(0.0, min(left, float(width)))
    right = max(0.0, min(right, float(width)))
    top = max(0.0, min(top, float(height)))
    bottom = max(0.0, min(bottom, float(height)))

    return [
        1000.0 * left / width,
        1000.0 * top / height,
        1000.0 * right / width,
        1000.0 * bottom / height,
    ]


def gt_people_as_normalized_prediction(
    frame: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a frame's GT people to the SAME compact schema that rollout inference
    feeds back as previous predictions.

    This function is used ONLY for real previous context frames.
    """
    path = Path(frame["_resolved_path"])
    width, height = get_image_size(path)

    people: List[Dict[str, Any]] = []

    for person in frame.get("people", [])[:MAX_PEOPLE]:
        bbox = normalize_pixel_box_to_1000(
            [float(v) for v in person["bbox"]],
            width,
            height,
        )

        people.append(
            {
                "person_ID": int(person["person_ID"]),
                "bbox": [round(v, 2) for v in bbox],
                "label": normalize_label(person["label"]),
            }
        )

    return {"people": people}


def target_answer_json(
    target_frame: Dict[str, Any],
) -> str:
    """
    Build the ONLY supervised answer: target-frame people in normalized [0,1000]
    xyxy coordinates.
    """
    target_path = Path(target_frame["_resolved_path"])
    width, height = get_image_size(target_path)

    people: List[Dict[str, Any]] = []

    for person in target_frame.get("people", [])[:MAX_PEOPLE]:
        bbox = normalize_pixel_box_to_1000(
            [float(v) for v in person["bbox"]],
            width,
            height,
        )

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue

        people.append(
            {
                "person_ID": int(person["person_ID"]),
                "bbox": [round(v, 2) for v in bbox],
                "label": normalize_label(person["label"]),
            }
        )

    # Compact JSON is closer to the format requested at inference.
    return json.dumps(
        {"people": people},
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ==============================================================================
# DATASET RECONSTRUCTION
# ==============================================================================

def split_matches(
    group_split: str,
    requested_split: str,
) -> bool:
    def normalize_split_name(value: str) -> str:
        normalized = value.strip().lower()
        aliases = {
            "validation": "val",
            "valid": "val",
            "dev": "val",
            "training": "train",
            "testing": "test",
        }
        return aliases.get(normalized, normalized)

    return (
        normalize_split_name(group_split)
        == normalize_split_name(requested_split)
    )


def load_video_timelines(
    requested_split: str,
    limit_videos: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Merge overlapping 16-frame JSON groups into one unique chronological timeline
    per source video for the requested video-level split.
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

        if not split_matches(str(group["split"]), requested_split):
            continue

        video_name = str(group["video_name"])
        dataset_name = dataset_name_from_json_file(
            json_path,
            video_name,
        )
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

            resolved_path = resolve_frame_path(
                frame["frame_path"],
                FRAME_ROOT,
                dataset_name,
                video_name,
            )

            frame_copy = dict(frame)
            frame_copy["_resolved_path"] = str(resolved_path)
            frame_copy["_source_json"] = json_path.name

            existing = video_entry["frames_by_number"].get(frame_number)

            if existing is None:
                video_entry["frames_by_number"][frame_number] = frame_copy
            else:
                old_name = windows_basename(existing["frame_path"])
                new_name = windows_basename(frame_copy["frame_path"])

                if old_name != new_name:
                    print(
                        f"[WARNING] {video_key} frame {frame_number} "
                        f"has differing paths: {old_name} vs {new_name}"
                    )

    final_videos: Dict[str, Dict[str, Any]] = {}

    for video_key, entry in videos.items():
        frames = [
            entry["frames_by_number"][frame_number]
            for frame_number in sorted(entry["frames_by_number"].keys())
        ]

        final_videos[video_key] = {
            "video_name": entry["video_name"],
            "source_dataset": entry["source_dataset"],
            "split": entry["split"],
            "frames": frames,
        }

    final_videos = dict(
        sorted(final_videos.items(), key=lambda x: x[0])
    )

    if limit_videos is not None:
        final_videos = dict(
            list(final_videos.items())[:limit_videos]
        )

    print(f"[DATA] JSON files found: {len(json_files)}")
    print(
        f"[DATA] Reconstructed {len(final_videos)} videos "
        f"for split={requested_split}"
    )

    total_targets = 0
    for key, video in final_videos.items():
        eligible_targets = max(0, len(video["frames"]) - 15)
        total_targets += eligible_targets
        print(
            f"       {key}: {len(video['frames'])} unique frames | "
            f"{eligible_targets} rollout targets"
        )

    print(
        f"[DATA] Eligible targets from frame 16 onward: {total_targets}"
    )

    return final_videos


def build_training_examples(
    videos: Dict[str, Dict[str, Any]],
    limit_samples: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Build rollout targets beginning at the 16th frame of each reconstructed video.

    Tuple:
        (video_key, zero_based_target_index)
    """
    examples: List[Tuple[str, int]] = []

    for video_key, video in videos.items():
        # zero-based index 15 == the 16th frame.
        for target_index in range(15, len(video["frames"])):
            examples.append((video_key, target_index))

    if limit_samples is not None:
        examples = examples[:limit_samples]

    return examples


# ==============================================================================
# TEACHER-FORCED TEMPORAL CONTEXT
# ==============================================================================

FIRST_PREDICTED_INDEX = 15  # zero-based index of frame 16


def build_teacher_forced_context(
    frames: List[Dict[str, Any]],
    target_index: int,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """
    Build exactly 15 REAL previous-frame context positions.

    Rollout starts at frame 16:
        target F16 -> F1..F15, all unannotated
        target F17 -> F2..F15 unannotated + F16 teacher-forced prediction
        target F18 -> F3..F15 unannotated + F16..F17 predictions
        ...
        target F31+ -> previous 15 frames all have teacher-forced predictions

    Ground truth is inserted ONLY where inference would contain an earlier model
    prediction. Pre-rollout context frames stay unannotated.
    """
    if target_index < FIRST_PREDICTED_INDEX:
        raise ValueError(
            "Rollout training begins at frame 16; "
            f"received zero-based target_index={target_index}."
        )

    history_indices = list(range(target_index - 15, target_index))

    context_paths: List[Path] = []
    context_metadata: List[Dict[str, Any]] = []

    for history_index in history_indices:
        history_frame = frames[history_index]
        history_number = int(history_frame["frame_number"])

        context_paths.append(
            Path(history_frame["_resolved_path"])
        )

        if history_index < FIRST_PREDICTED_INDEX:
            # At inference these frames were observed before rollout prediction
            # began, so there is genuinely no earlier prediction to feed back.
            context_metadata.append(
                {
                    "kind": "unannotated_context",
                    "dataset_frame_number": history_number,
                    "prediction": None,
                }
            )
        else:
            # At inference this frame would carry the model's own earlier
            # prediction. During training, use GT as teacher forcing.
            context_metadata.append(
                {
                    "kind": "previous_prediction",
                    "dataset_frame_number": history_number,
                    "prediction": gt_people_as_normalized_prediction(
                        history_frame
                    ),
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
    Match rollout inference semantics while using teacher forcing only for frames
    that would already have a previous model prediction at inference time.
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
        "Positions 1-15 are the 15 real previous frames; position 16 is the target."
    )
    lines.append("")
    lines.append("CONTEXT POSITION METADATA:")

    for position, metadata in enumerate(
        context_metadata,
        start=1,
    ):
        if metadata["kind"] == "unannotated_context":
            lines.append(
                f"Position {position}: previous dataset frame "
                f"{metadata['dataset_frame_number']}; "
                "NO MODEL PREDICTION is available because this frame occurred "
                "before rollout predictions began."
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


def build_input_messages(
    video: Dict[str, Any],
    target_frame: Dict[str, Any],
    context_paths: List[Path],
    context_metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    target_path = Path(target_frame["_resolved_path"])
    model_paths = context_paths + [target_path]

    context_text = build_context_text(
        video,
        target_frame,
        context_metadata,
    )

    # qwen-vl-utils reads video resizing constraints from the visual content item
    # before the processor call. This is the effective place to cap video pixels.
    video_content: Dict[str, Any] = {
        "type": "video",
        "video": [
            str(path.resolve())
            for path in model_paths
        ],
    }

    if MAX_PIXELS is not None:
        video_content["max_pixels"] = MAX_PIXELS

    if MIN_PIXELS is not None:
        video_content["min_pixels"] = MIN_PIXELS

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
                video_content,
                {
                    "type": "text",
                    "text": context_text,
                },
            ],
        },
    ]


# ==============================================================================
# MODEL + LoRA
# ==============================================================================

def load_model_and_processor(
    dtype: torch.dtype,
    resume_adapter_path: Optional[Path] = None,
):
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

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Match the visual-resolution configuration of the attached trainer.
    if MAX_PIXELS is not None and hasattr(
        processor.image_processor,
        "max_pixels",
    ):
        processor.image_processor.max_pixels = MAX_PIXELS

    if MIN_PIXELS is not None and hasattr(
        processor.image_processor,
        "min_pixels",
    ):
        processor.image_processor.min_pixels = MIN_PIXELS

    model.config.use_cache = False

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

        # Needed by PEFT + gradient checkpointing on many transformer stacks.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    if resume_adapter_path is not None:
        print(f"[TRAIN] Loading adapter weights from {resume_adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            resume_adapter_path,
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            model,
            lora_config,
        )

    model.train()
    model.print_trainable_parameters()

    print(f"[MODEL] loaded on {model.device}")

    return model, processor


# ==============================================================================
# MULTIMODAL SUPERVISED EXAMPLE CONSTRUCTION
# ==============================================================================

def prepare_training_tensors(
    processor,
    video: Dict[str, Any],
    target_frame: Dict[str, Any],
    context_paths: List[Path],
    context_metadata: List[Dict[str, Any]],
) -> Tuple[Dict[str, torch.Tensor], str, int]:
    """
    Build the multimodal input and labels.

    Loss masking:
        - system prompt: ignored
        - 16 visual positions: ignored
        - user/context text: ignored
        - assistant target JSON: SUPERVISED

    The multimodal video is processed only once. The assistant boundary is
    obtained from cheap text-only tokenization because the visual placeholder
    expansion is identical in the prompt and full conversation.
    """
    input_messages = build_input_messages(
        video=video,
        target_frame=target_frame,
        context_paths=context_paths,
        context_metadata=context_metadata,
    )

    answer_json = target_answer_json(target_frame)

    full_messages = input_messages + [
        {
            "role": "assistant",
            "content": answer_json,
        }
    ]

    prompt_text = processor.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Determine how many TEXT tokens belong to the supervised assistant suffix.
    # This avoids a second expensive multimodal processor pass.
    prompt_text_ids = processor.tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    full_text_ids = processor.tokenizer(
        full_text,
        add_special_tokens=False,
    )["input_ids"]

    if full_text_ids[:len(prompt_text_ids)] != prompt_text_ids:
        raise RuntimeError(
            "Assistant masking failed: prompt tokenization is not an exact "
            "prefix of the full conversation tokenization."
        )

    assistant_suffix_tokens = len(full_text_ids) - len(prompt_text_ids)
    if assistant_suffix_tokens <= 0:
        raise RuntimeError(
            f"No assistant suffix tokens found. Target JSON was: {answer_json}"
        )

    (
        image_inputs,
        video_inputs,
        video_kwargs,
    ) = process_vision_info(
        input_messages,
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

    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=videos,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )

    full_length = int(full_inputs["input_ids"].shape[1])
    prompt_length = full_length - assistant_suffix_tokens

    if prompt_length <= 0 or prompt_length >= full_length:
        raise RuntimeError(
            f"Assistant masking failed: prompt_length={prompt_length}, "
            f"full_length={full_length}, "
            f"assistant_suffix_tokens={assistant_suffix_tokens}. "
            f"Target JSON was: {answer_json}"
        )

    labels = full_inputs["input_ids"].clone()

    # Supervise assistant answer only (including its closing chat token, matching
    # the previous implementation's behavior).
    labels[:, :prompt_length] = -100

    # Defensive pad masking.
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[
            full_inputs["input_ids"] == pad_token_id
        ] = -100

    full_inputs["labels"] = labels

    supervised_tokens = int(
        (labels != -100).sum().item()
    )

    if supervised_tokens <= 0:
        raise RuntimeError(
            f"No supervised tokens found for target "
            f"{target_frame['frame_number']}."
        )

    return full_inputs, answer_json, supervised_tokens


def move_batch_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(
                device,
                non_blocking=True,
            )
        else:
            moved[key] = value

    return moved


# ==============================================================================
# CHECKPOINTING / LOGGING
# ==============================================================================

def find_last_adapter_checkpoint(
    output_dir: Path,
) -> Optional[Path]:
    if not output_dir.exists():
        return None

    checkpoints: List[Tuple[int, Path]] = []

    for path in output_dir.glob("checkpoint-step-*"):
        if not path.is_dir():
            continue

        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue

        # Only checkpoints created by the corrected trainer contain enough
        # state for a true optimizer/scheduler/scaler resume.
        if not (path / "trainer_state.pt").exists():
            continue

        checkpoints.append((step, path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def prune_old_checkpoints(output_dir: Path) -> None:
    if SAVE_TOTAL_LIMIT is None or SAVE_TOTAL_LIMIT <= 0:
        return

    checkpoints: List[Tuple[int, Path]] = []

    for path in output_dir.glob("checkpoint-step-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue

        # Legacy adapter-only checkpoints from the previous trainer are ignored
        # here; only fully resumable checkpoints count toward retention.
        if not (path / "trainer_state.pt").exists():
            continue

        checkpoints.append((step, path))

    checkpoints.sort(key=lambda item: item[0])

    while len(checkpoints) > SAVE_TOTAL_LIMIT:
        _, old_path = checkpoints.pop(0)
        import shutil
        shutil.rmtree(old_path, ignore_errors=True)
        print(f"[SAVE] Removed old checkpoint: {old_path}")



def save_adapter_checkpoint(
    model,
    processor,
    optimizer,
    scheduler,
    scaler,
    optimizer_step: int,
    epoch: int,
    micro_step_in_epoch: int,
    global_micro_step: int,
) -> Path:
    checkpoint_dir = (
        OUTPUT_DIR
        / f"checkpoint-step-{optimizer_step:08d}"
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    trainer_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "micro_step_in_epoch": micro_step_in_epoch,
        "optimizer_step": optimizer_step,
        "global_micro_step": global_micro_step,
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    torch.save(
        trainer_state,
        checkpoint_dir / "trainer_state.pt",
    )

    print(f"\n[SAVE] Full training checkpoint: {checkpoint_dir}")
    prune_old_checkpoints(OUTPUT_DIR)

    return checkpoint_dir


def load_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> Dict[str, int]:
    """Restore optimizer/scheduler/scaler/RNG and exact training position."""
    state_path = checkpoint_dir / "trainer_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_dir} has adapter weights but no "
            "trainer_state.pt, so exact resume is not possible."
        )

    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )

    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])

    # Make sure optimizer tensors follow the model onto the active GPU.
    for optimizer_state in optimizer.state.values():
        for key, value in list(optimizer_state.items()):
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)

    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

    return {
        "epoch": int(state["epoch"]),
        "micro_step_in_epoch": int(state["micro_step_in_epoch"]),
        "optimizer_step": int(state["optimizer_step"]),
        "global_micro_step": int(state["global_micro_step"]),
    }


def save_training_config() -> None:
    config = {
        "model_name": MODEL_NAME,
        "split": SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": SEED,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "eval_every_optimizer_steps": EVAL_EVERY_OPTIMIZER_STEPS,
        "eval_max_samples": EVAL_MAX_SAMPLES,
        "resume_from_last_checkpoint": RESUME_FROM_LAST_CHECKPOINT,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "max_pixels": MAX_PIXELS,
        "min_pixels": MIN_PIXELS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "temporal_method": "teacher_forced_rollout_starting_at_frame_16",
        "first_predicted_frame": 16,
        "context_length_frames": 15,
        "target_position": 16,
        "loss_scope": "assistant_target_json_only",
        "bbox_coordinate_space": "normalized_0_1000_xyxy",
    }

    with (OUTPUT_DIR / "training_config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )


def append_loss_log(record: Dict[str, Any]) -> None:
    path = OUTPUT_DIR / "training_log.jsonl"

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def append_validation_log(record: Dict[str, Any]) -> None:
    path = OUTPUT_DIR / "validation_log.jsonl"

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def evaluate_validation_loss(
    model,
    processor,
    validation_videos: Dict[str, Dict[str, Any]],
    validation_examples: List[Tuple[str, int]],
    dtype: torch.dtype,
    optimizer_step: int,
) -> Optional[float]:
    """
    Lightweight teacher-forced validation on a fixed small subset.

    This intentionally measures supervised target-token loss rather than running
    autoregressive generation, keeping the runtime overhead small. Full rollout
    detection/classification metrics should still be computed with the separate
    inference/evaluation pipeline after training/checkpoint selection.
    """
    if not validation_examples:
        return None

    model.eval()
    losses: List[float] = []

    progress = tqdm(
        validation_examples,
        desc=f"Validation @ step {optimizer_step}",
        leave=False,
    )

    try:
        with torch.inference_mode():
            for video_key, target_index in progress:
                video = validation_videos[video_key]
                frames = video["frames"]
                target_frame = frames[target_index]

                context_paths, context_metadata = (
                    build_teacher_forced_context(
                        frames=frames,
                        target_index=target_index,
                    )
                )

                batch, _, _ = prepare_training_tensors(
                    processor=processor,
                    video=video,
                    target_frame=target_frame,
                    context_paths=context_paths,
                    context_metadata=context_metadata,
                )
                batch = move_batch_to_device(
                    batch,
                    model.device,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=True,
                ):
                    outputs = model(**batch)
                    loss_value = float(
                        outputs.loss.detach().float().item()
                    )

                losses.append(loss_value)
                progress.set_postfix(loss=f"{loss_value:.4f}")

                del batch, outputs
    finally:
        model.train()

    if not losses:
        return None

    validation_loss = sum(losses) / len(losses)
    append_validation_log(
        {
            "optimizer_step": optimizer_step,
            "validation_loss": validation_loss,
            "num_examples": len(losses),
        }
    )
    print(
        f"\n[EVAL] step={optimizer_step} | "
        f"validation_loss={validation_loss:.4f} | "
        f"examples={len(losses)}"
    )
    return validation_loss


def train(
    model,
    processor,
    videos: Dict[str, Dict[str, Any]],
    examples: List[Tuple[str, int]],
    validation_videos: Dict[str, Dict[str, Any]],
    validation_examples: List[Tuple[str, int]],
    dtype: torch.dtype,
    resume_checkpoint_path: Optional[Path] = None,
) -> None:
    if not examples:
        raise RuntimeError(
            f"No training examples found for SPLIT={SPLIT}."
        )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(
        len(examples)
        / GRADIENT_ACCUMULATION_STEPS
    )
    total_optimizer_steps = (
        EPOCHS * updates_per_epoch
    )
    warmup_steps = min(WARMUP_STEPS, total_optimizer_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    use_grad_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_grad_scaler,
    )

    start_epoch = 1
    resume_micro_step_in_epoch = 0
    optimizer_step = 0
    global_micro_step = 0

    if resume_checkpoint_path is not None:
        resume_state = load_training_state(
            checkpoint_dir=resume_checkpoint_path,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=model.device,
        )
        start_epoch = resume_state["epoch"]
        resume_micro_step_in_epoch = resume_state["micro_step_in_epoch"]
        optimizer_step = resume_state["optimizer_step"]
        global_micro_step = resume_state["global_micro_step"]

        # A checkpoint saved at the final micro-step of an epoch should continue
        # at the next epoch rather than replaying the completed one.
        if resume_micro_step_in_epoch >= len(examples):
            start_epoch += 1
            resume_micro_step_in_epoch = 0

        print(
            f"[TRAIN] Exact resume: epoch={start_epoch}, "
            f"next_micro_step={resume_micro_step_in_epoch + 1}, "
            f"optimizer_step={optimizer_step}, "
            f"global_micro_step={global_micro_step}"
        )

    print(
        f"[TRAIN] examples={len(examples)} | "
        f"epochs={EPOCHS} | "
        f"grad_accum={GRADIENT_ACCUMULATION_STEPS} | "
        f"optimizer_steps={total_optimizer_steps} | "
        f"warmup_steps={warmup_steps} | "
        f"validation_examples={len(validation_examples)}"
    )

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    running_micro_steps = 0

    if start_epoch > EPOCHS:
        print("[TRAIN] Checkpoint already completed all configured epochs.")
    else:
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_examples = list(examples)

            if SHUFFLE_EACH_EPOCH:
                epoch_rng = random.Random(SEED + epoch)
                epoch_rng.shuffle(epoch_examples)

            start_index = (
                resume_micro_step_in_epoch
                if epoch == start_epoch
                else 0
            )

            progress = tqdm(
                enumerate(
                    epoch_examples[start_index:],
                    start=start_index + 1,
                ),
                total=len(epoch_examples),
                initial=start_index,
                desc=f"Epoch {epoch}/{EPOCHS}",
            )

            for micro_step_in_epoch, (
                video_key,
                target_index,
            ) in progress:
                global_micro_step += 1

                video = videos[video_key]
                frames = video["frames"]
                target_frame = frames[target_index]

                context_paths, context_metadata = (
                    build_teacher_forced_context(
                        frames=frames,
                        target_index=target_index,
                    )
                )

                try:
                    batch, answer_json, supervised_tokens = (
                        prepare_training_tensors(
                            processor=processor,
                            video=video,
                            target_frame=target_frame,
                            context_paths=context_paths,
                            context_metadata=context_metadata,
                        )
                    )

                    batch = move_batch_to_device(
                        batch,
                        model.device,
                    )

                    with torch.autocast(
                        device_type="cuda",
                        dtype=dtype,
                        enabled=True,
                    ):
                        outputs = model(**batch)
                        raw_loss = outputs.loss

                        loss = (
                            raw_loss
                            / GRADIENT_ACCUMULATION_STEPS
                        )

                    scaler.scale(loss).backward()

                    loss_value = float(
                        raw_loss.detach().float().item()
                    )
                    running_loss += loss_value
                    running_micro_steps += 1

                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()

                    raise RuntimeError(
                        "\nCUDA OOM while training "
                        f"{video_key} / frame "
                        f"{target_frame['frame_number']}.\n"
                        "The example contains 16 visual positions. "
                        "Reduce MAX_PIXELS in BOTH training and inference, "
                        "use gradient checkpointing/a smaller model, or use "
                        "the 128 GB Thor."
                    )

                should_update = (
                    micro_step_in_epoch
                    % GRADIENT_ACCUMULATION_STEPS
                    == 0
                )

                is_last_micro_step = (
                    micro_step_in_epoch
                    == len(epoch_examples)
                )

                if should_update or is_last_micro_step:
                    remainder = (
                        micro_step_in_epoch
                        % GRADIENT_ACCUMULATION_STEPS
                    )

                    if (
                        is_last_micro_step
                        and remainder != 0
                    ):
                        correction = (
                            GRADIENT_ACCUMULATION_STEPS
                            / remainder
                        )
                        for parameter in trainable_parameters:
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)

                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        MAX_GRAD_NORM,
                    )

                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    optimizer_step += 1

                    average_running_loss = (
                        running_loss
                        / max(running_micro_steps, 1)
                    )

                    current_lr = float(
                        scheduler.get_last_lr()[0]
                    )

                    progress.set_postfix(
                        loss=f"{average_running_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                        opt_step=optimizer_step,
                    )

                    if (
                        optimizer_step
                        % LOG_EVERY_OPTIMIZER_STEPS
                        == 0
                    ):
                        log_record = {
                            "epoch": epoch,
                            "optimizer_step": optimizer_step,
                            "global_micro_step": global_micro_step,
                            "loss": average_running_loss,
                            "learning_rate": current_lr,
                            "video_key": video_key,
                            "target_frame_number": int(
                                target_frame["frame_number"]
                            ),
                            "supervised_tokens_last_example": (
                                supervised_tokens
                            ),
                        }
                        append_loss_log(log_record)

                        running_loss = 0.0
                        running_micro_steps = 0

                    if (
                        EVAL_EVERY_OPTIMIZER_STEPS
                        and validation_examples
                        and optimizer_step
                        % EVAL_EVERY_OPTIMIZER_STEPS
                        == 0
                    ):
                        evaluate_validation_loss(
                            model=model,
                            processor=processor,
                            validation_videos=validation_videos,
                            validation_examples=validation_examples,
                            dtype=dtype,
                            optimizer_step=optimizer_step,
                        )

                    if (
                        SAVE_EVERY_OPTIMIZER_STEPS
                        and optimizer_step
                        % SAVE_EVERY_OPTIMIZER_STEPS
                        == 0
                    ):
                        save_adapter_checkpoint(
                            model=model,
                            processor=processor,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            optimizer_step=optimizer_step,
                            epoch=epoch,
                            micro_step_in_epoch=micro_step_in_epoch,
                            global_micro_step=global_micro_step,
                        )

                # Release references normally; do NOT force Python GC or empty the
                # CUDA caching allocator on every sample because both are costly.
                del batch
                if "outputs" in locals():
                    del outputs
                if "raw_loss" in locals():
                    del raw_loss
                if "loss" in locals():
                    del loss

            resume_micro_step_in_epoch = 0

            # Save an exact resumable checkpoint at every epoch boundary, even if
            # the final step was not a multiple of SAVE_EVERY_OPTIMIZER_STEPS.
            save_adapter_checkpoint(
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                optimizer_step=optimizer_step,
                epoch=epoch,
                micro_step_in_epoch=len(epoch_examples),
                global_micro_step=global_micro_step,
            )

            epoch_dir = OUTPUT_DIR / f"epoch-{epoch:02d}"
            epoch_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            model.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)

            print(f"\n[SAVE] End-of-epoch adapter: {epoch_dir}")

    final_adapter_dir = OUTPUT_DIR / "final_adapter"
    final_adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)

    print(f"\n[SAVE] Final adapter: {final_adapter_dir}")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(SEED)
    save_training_config()

    dtype = check_cuda()

    videos = load_video_timelines(
        requested_split=SPLIT,
        limit_videos=LIMIT_VIDEOS,
    )
    examples = build_training_examples(
        videos,
        limit_samples=LIMIT_TRAIN_SAMPLES,
    )

    print(
        f"[DATA] Training examples after limits: "
        f"{len(examples)}"
    )

    validation_videos = load_video_timelines(
        requested_split=VALIDATION_SPLIT,
        limit_videos=LIMIT_VIDEOS,
    )
    all_validation_examples = build_training_examples(
        validation_videos,
        limit_samples=None,
    )

    if (
        EVAL_MAX_SAMPLES is not None
        and EVAL_MAX_SAMPLES > 0
        and len(all_validation_examples) > EVAL_MAX_SAMPLES
    ):
        validation_rng = random.Random(SEED + 100_000)
        validation_examples = validation_rng.sample(
            all_validation_examples,
            EVAL_MAX_SAMPLES,
        )
    else:
        validation_examples = all_validation_examples

    print(
        f"[DATA] Fixed validation subset: "
        f"{len(validation_examples)} examples"
    )

    resume_checkpoint_path = None
    if RESUME_FROM_LAST_CHECKPOINT:
        candidate = find_last_adapter_checkpoint(
            OUTPUT_DIR
        )
        if candidate is not None:
            resume_checkpoint_path = candidate
            print(
                f"[TRAIN] Resuming full training state from "
                f"{resume_checkpoint_path}"
            )
        else:
            print(
                "[TRAIN] No complete checkpoint found; starting from "
                "base model + new LoRA adapter."
            )

    model, processor = load_model_and_processor(
        dtype,
        resume_adapter_path=resume_checkpoint_path,
    )

    train(
        model=model,
        processor=processor,
        videos=videos,
        examples=examples,
        validation_videos=validation_videos,
        validation_examples=validation_examples,
        dtype=dtype,
        resume_checkpoint_path=resume_checkpoint_path,
    )

    print("\n[DONE]")
    print(f"Training output: {OUTPUT_DIR}")
    print(f"Final adapter: {OUTPUT_DIR / 'final_adapter'}")


if __name__ == "__main__":
    main()
