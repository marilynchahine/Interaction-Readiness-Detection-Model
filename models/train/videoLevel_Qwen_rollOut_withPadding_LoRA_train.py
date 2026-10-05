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

Training uses TEACHER FORCING:
    - real previous context frames receive their ground-truth annotations,
    - those annotations are formatted exactly like inference-time previous
      predictions ("MODEL PREDICTION = ..."),
    - warm-up padding is identical to inference,
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
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/"
    "outputs/videoLevel_fineTuned_rollout/Qwen3VL_2B_Instruct"
)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# Training data only.
SPLIT = "Train"

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
EPOCHS = 2
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

# Match the previous trainer's validation/checkpoint cadence.
EVAL_EVERY_OPTIMIZER_STEPS = 200

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
    return (
        group_split.strip().lower()
        == requested_split.strip().lower()
    )


def load_video_timelines() -> Dict[str, Dict[str, Any]]:
    """
    Merge overlapping 16-frame JSON groups into one unique chronological timeline
    per source video, just like rollout inference.
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

    if LIMIT_VIDEOS is not None:
        final_videos = dict(
            list(final_videos.items())[:LIMIT_VIDEOS]
        )

    print(f"[DATA] JSON files found: {len(json_files)}")
    print(
        f"[DATA] Reconstructed {len(final_videos)} videos "
        f"for split={SPLIT}"
    )

    total_frames = 0
    for key, video in final_videos.items():
        total_frames += len(video["frames"])
        print(f"       {key}: {len(video['frames'])} unique frames")

    print(f"[DATA] Unique target-frame training examples: {total_frames}")

    return final_videos


def build_training_examples(
    videos: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """
    Each unique frame becomes one target example.

    Tuple:
        (video_key, zero_based_target_index)
    """
    examples: List[Tuple[str, int]] = []

    for video_key, video in videos.items():
        for target_index in range(len(video["frames"])):
            examples.append((video_key, target_index))

    if LIMIT_TRAIN_SAMPLES is not None:
        examples = examples[:LIMIT_TRAIN_SAMPLES]

    return examples


# ==============================================================================
# TEACHER-FORCED TEMPORAL CONTEXT
# ==============================================================================

def build_teacher_forced_context(
    frames: List[Dict[str, Any]],
    target_index: int,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """
    Build exactly 15 context positions.

    Warm-up is IDENTICAL to rollout inference:
        target 1 -> [F1] * 15 as unannotated target-padding copies
        target 2 -> [F1 + GT1] + [F2] * 14
        ...
        target 16 -> [F1 + GT1, ..., F15 + GT15]

    For target >= 17, use the previous 15 real frames.

    The only difference from inference is that previous annotations are GT
    (teacher forcing) rather than generated predictions.
    """
    target_frame = frames[target_index]
    target_path = Path(target_frame["_resolved_path"])

    history_start = max(0, target_index - 15)
    history_indices = list(range(history_start, target_index))

    context_paths: List[Path] = []
    context_metadata: List[Dict[str, Any]] = []

    for history_index in history_indices:
        history_frame = frames[history_index]
        history_number = int(history_frame["frame_number"])

        context_paths.append(
            Path(history_frame["_resolved_path"])
        )
        context_metadata.append(
            {
                "kind": "previous_frame",
                "dataset_frame_number": history_number,

                # IMPORTANT:
                # This key and schema intentionally match inference-time context.
                "prediction": gt_people_as_normalized_prediction(
                    history_frame
                ),
            }
        )

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
    Intentionally identical in wording to rollout inference.

    Although training values are teacher-forced GT, they are presented through
    the same "MODEL PREDICTION" interface. This minimizes train/inference prompt
    distribution shift.
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

    Returns:
        model_inputs
        target_json
        number_of_supervised_tokens
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

    # Process full conversation once.
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

    # Process the prompt-only prefix with the SAME visual inputs. We need its
    # expanded multimodal token length so we can mask everything before the
    # assistant answer.
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=videos,
        video_metadata=video_metadatas,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )

    prompt_length = int(prompt_inputs["input_ids"].shape[1])
    full_length = int(full_inputs["input_ids"].shape[1])

    if prompt_length >= full_length:
        raise RuntimeError(
            f"Assistant masking failed: prompt_length={prompt_length}, "
            f"full_length={full_length}. Target JSON was: {answer_json}"
        )

    labels = full_inputs["input_ids"].clone()

    # Supervise assistant answer only.
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

    del prompt_inputs

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
    optimizer_step: int,
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

    print(f"\n[SAVE] Adapter checkpoint: {checkpoint_dir}")
    prune_old_checkpoints(OUTPUT_DIR)

    return checkpoint_dir


def save_training_config() -> None:
    config = {
        "model_name": MODEL_NAME,
        "split": SPLIT,
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
        "resume_from_last_checkpoint": RESUME_FROM_LAST_CHECKPOINT,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "max_pixels": MAX_PIXELS,
        "min_pixels": MIN_PIXELS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "temporal_method": "teacher_forced_prediction_conditioned_rollout",
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


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train(
    model,
    processor,
    videos: Dict[str, Dict[str, Any]],
    examples: List[Tuple[str, int]],
    dtype: torch.dtype,
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

    print(
        f"[TRAIN] examples={len(examples)} | "
        f"epochs={EPOCHS} | "
        f"grad_accum={GRADIENT_ACCUMULATION_STEPS} | "
        f"optimizer_steps={total_optimizer_steps} | "
        f"warmup_steps={warmup_steps}"
    )

    optimizer.zero_grad(set_to_none=True)

    optimizer_step = 0
    global_micro_step = 0
    running_loss = 0.0
    running_micro_steps = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_examples = list(examples)

        if SHUFFLE_EACH_EPOCH:
            epoch_rng = random.Random(SEED + epoch)
            epoch_rng.shuffle(epoch_examples)

        progress = tqdm(
            enumerate(epoch_examples, start=1),
            total=len(epoch_examples),
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

                # Mixed precision follows the base model dtype.
                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=True,
                ):
                    outputs = model(**batch)
                    raw_loss = outputs.loss

                    # Normalize for gradient accumulation.
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
                    "On a smaller GPU, reduce visual resolution in BOTH "
                    "training and inference, use a smaller Qwen3-VL model, "
                    "or run on the 128 GB Thor."
                )

            should_update = (
                micro_step_in_epoch
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            )

            # Flush the final partial accumulation at end of epoch.
            is_last_micro_step = (
                micro_step_in_epoch
                == len(epoch_examples)
            )

            if should_update or is_last_micro_step:
                # If this is a partial final accumulation, its gradients were
                # divided by the full configured accumulation count. Correct
                # the scale so it behaves like the actual partial batch size.
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
                    SAVE_EVERY_OPTIMIZER_STEPS
                    and optimizer_step
                    % SAVE_EVERY_OPTIMIZER_STEPS
                    == 0
                ):
                    save_adapter_checkpoint(
                        model=model,
                        processor=processor,
                        optimizer_step=optimizer_step,
                    )

            # Release the very large multimodal tensors immediately.
            del batch
            if "outputs" in locals():
                del outputs
            if "raw_loss" in locals():
                del raw_loss
            if "loss" in locals():
                del loss

            gc.collect()
            torch.cuda.empty_cache()

        # Epoch-end adapter.
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

    videos = load_video_timelines()
    examples = build_training_examples(videos)

    print(
        f"[DATA] Training examples after limits: "
        f"{len(examples)}"
    )

    resume_adapter_path = None
    if RESUME_FROM_LAST_CHECKPOINT:
        resume_adapter_path = find_last_adapter_checkpoint(
            OUTPUT_DIR
        )
        if resume_adapter_path is not None:
            print(
                f"[TRAIN] Resuming adapter weights from "
                f"{resume_adapter_path}"
            )
            print(
                "[TRAIN] Note: this custom training loop resumes LoRA weights, "
                "not optimizer/scheduler state."
            )
        else:
            print(
                "[TRAIN] No adapter checkpoint found; starting from "
                "base model + new LoRA adapter."
            )

    model, processor = load_model_and_processor(
        dtype,
        resume_adapter_path=resume_adapter_path,
    )

    train(
        model=model,
        processor=processor,
        videos=videos,
        examples=examples,
        dtype=dtype,
    )

    print("\n[DONE]")
    print(f"Training output: {OUTPUT_DIR}")
    print(f"Final adapter: {OUTPUT_DIR / 'final_adapter'}")


if __name__ == "__main__":
    main()
