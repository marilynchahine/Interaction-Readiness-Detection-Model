#!/usr/bin/env python3
"""
LoRA supervised fine-tuning for temporal person detection + interaction readiness
with Qwen3-VL.

CURRENT TEMPORAL TASK
---------------------
Each JSON file is one 16-frame sliding window.

Frames 1-15:
    - raw images are shown to the model as temporal context
    - NO person IDs, bounding boxes, tracking annotations, or
      interaction-readiness labels are supplied

Frame 16:
    - raw image is shown to the model
    - NO frame-16 annotations are included in the prompt
    - frame-16 ground truth is used ONLY as the supervised assistant target
    - the model learns to:
        1. detect visible humans
        2. select up to 5 people
        3. predict a bbox for every selected person
        4. predict an interaction-readiness label for every selected person

SFT FORMULATION
---------------
SYSTEM:
    Same task definition as the current zero-shot experiment.

USER:
    16 raw frames
    + annotation-free temporal instructions

ASSISTANT TARGET:
    Frame-16 ground-truth detections only:
    {
      "people": [
        {
          "person_ID": 1,
          "bbox": [x1,y1,x2,y2],   # normalized 0..1000
          "label": "interaction_ready"
        }
      ]
    }

The model therefore receives the SAME INFORMATION at fine-tuning and inference
time. Ground-truth annotations from frames 1-15 are never exposed.

Only Train groups update weights.
Val groups are used for validation.
Test groups are untouched.

Loss is masked on the prompt and computed only on assistant target tokens.

GPU-only.
"""

import gc
import inspect
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Helps CUDA allocator fragmentation on supported PyTorch versions.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info


# ==============================================================================
# USER CONFIGURATION — EDIT THESE VALUES
# ==============================================================================

JSON_DIR = Path("/home/marilyn/Downloads/data/JSON_temporal")

# Set to None if frame_path values inside the JSON already work on this machine.
# Otherwise point this to the root of your frames directory.
FRAME_ROOT = Path("/home/marilyn/Downloads/data/frames")

OUTPUT_DIR = Path(
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/videoLevel_fineTuned/Qwen3VL_2B_Instruct"
)

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# ---------------- LoRA ----------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# ---------------- Training ----------------

NUM_TRAIN_EPOCHS = 2

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-6
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

LOGGING_STEPS = 10
EVAL_STEPS = 200
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

# Set e.g. 20 for pipeline sanity testing.
MAX_TRAIN_GROUPS = None
MAX_VAL_GROUPS = None

SEED = 42

# "auto", "flash_attention_2", "sdpa", or "eager"
ATTN_IMPLEMENTATION = "auto"

GRADIENT_CHECKPOINTING = False
RESUME_FROM_LAST_CHECKPOINT = True

DATALOADER_NUM_WORKERS = 0

# Optional processor pixel limits.
# Leave as None on a large-memory GPU such as your 128 GB Thor.
MAX_PIXELS = 448 * 448
MIN_PIXELS = None

# ==============================================================================


CLASS_NAMES = {
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
}


# IMPORTANT:
# Keep this prompt synchronized with the zero-shot inference script.
SYSTEM_PROMPT = """You are an expert in human-robot interaction, person detection, visual grounding, and temporal human behavior analysis.

You will receive an ordered sequence of 16 raw video frames.

TEMPORAL STRUCTURE

* Frames 1 through 15 are CONTEXT frames.
* Frame 16 is the TARGET frame.
* Frames 1-15 are used only as raw visual-temporal context.
* Output predictions ONLY for frame 16.

Your task has THREE stages:

STEP 1 — DETECT PEOPLE IN FRAME 16

First inspect frame 16 and detect the visible humans.

Detection must be based on the visual content of frame 16 itself.

If at least one human is visible in frame 16, you MUST return at least one person.

Do not omit a person because they:

* appear uninterested,
* are facing away,
* are not looking at the robot,
* are not interaction-ready,
* are partially occluded,
* or appear to be only passing through the scene.

Interaction-readiness must NOT be used to decide whether a human is detected.

STEP 2 — SELECT UP TO 5 PEOPLE

After detecting the visible humans:

* If 5 or fewer humans are visible, return all of them.
* If more than 5 humans are visible, select exactly 5.

When more than 5 humans are visible, prioritize the people who are most relevant to the robot based primarily on physical and visual proximity.

Rank people using the following criteria, in this order:

1. apparent closeness to the robot/camera,
2. apparent size of the person in frame 16,
3. how clearly visible and prominent the person is.

Prefer nearby, large, clearly visible people over distant background pedestrians.

Do NOT use interaction-readiness as the main criterion for selecting people.

STEP 3 — CLASSIFY EACH SELECTED PERSON

For EVERY selected person, predict exactly one interaction-readiness label:

* not_interaction_ready:
  The person is not currently signaling readiness or intention to initiate an interaction with the robot.

* interaction_ready:
  The person is signaling readiness or intention to initiate an interaction with the robot, but the interaction itself has not yet started.

* interaction_ongoing:
  The person is already actively engaged in an interaction with the robot.

Use frames 1-15 as temporal context to help classify the people detected in frame 16.

Relevant temporal cues may include:

* approaching or moving away from the robot,
* changes in distance,
* body orientation,
* head direction or gaze when visible,
* stopping or waiting near the robot,
* gestures,
* posture changes,
* repeated attention toward the robot,
* and evidence that an interaction has already started.

IMPORTANT: DETECTION COMES BEFORE CLASSIFICATION

First determine which humans are visible in frame 16.

Then select up to 5 people based mainly on proximity and visual prominence.

Only after detection and selection should you predict each person's interaction-readiness state.

A selected person classified as not_interaction_ready is still a valid detection and MUST be returned.

BOUNDING-BOX OUTPUT

For every selected person in frame 16:

* bbox format: [x1, y1, x2, y2]
* coordinates must be normalized to 0-1000 relative to frame 16
* 0 corresponds to the left/top image edge
* 1000 corresponds to the right/bottom image edge
* ensure x2 > x1 and y2 > y1
* make the box tightly cover the visible person

PERSON IDS

Assign a unique person_ID to each returned person.

Use consecutive integer IDs starting from 1:
1, 2, 3, 4, 5.

OUTPUT REQUIREMENTS

Every returned person MUST contain all three fields:

* person_ID
* bbox
* label

Return each selected person exactly once.

Return ONLY valid JSON.
Do not include markdown, prose, explanations, comments, or additional fields.

Required output schema:

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

EMPTY-FRAME RULE

Return {"people":[]} ONLY if there are literally zero visible humans anywhere in frame 16.

If at least one human is visible in frame 16, returning an empty list is incorrect.
"""


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
    """
    Works with:
      - already-valid Windows paths,
      - already-valid Linux paths,
      - JSONs copied between machines with FRAME_ROOT supplied.
    """
    direct = Path(original_path)

    if direct.exists():
        return direct.resolve()

    if frame_root is None:
        raise FileNotFoundError(
            f"Frame path does not exist on this machine: {original_path}\n"
            "Set FRAME_ROOT if these JSONs were generated on another machine."
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


def clamp_box(
    box: List[float],
    width: int,
    height: int,
) -> List[float]:
    x1, y1, x2, y2 = map(float, box)

    x1 = max(0.0, min(x1, float(width)))
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))

    return [x1, y1, x2, y2]


def normalize_box_1000(
    box: List[float],
    width: int,
    height: int,
) -> List[int]:
    x1, y1, x2, y2 = clamp_box(
        box,
        width,
        height,
    )

    return [
        round(1000.0 * x1 / width),
        round(1000.0 * y1 / height),
        round(1000.0 * x2 / width),
        round(1000.0 * y2 / height),
    ]


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def prepare_group_images(
    group: Dict[str, Any],
    json_path: Path,
) -> Tuple[List[Path], List[Path], Path]:
    """
    Return all 16 raw paths.

    No context-frame annotations are drawn.
    No context-frame annotations are converted to text.
    """
    if len(group["frames"]) != 16:
        raise ValueError(
            f"{json_path.name}: expected exactly 16 frames, "
            f"found {len(group['frames'])}."
        )

    dataset_name = dataset_name_from_json_file(
        json_path,
        group["video_name"],
    )

    original_paths: List[Path] = []

    for frame in group["frames"]:
        source = resolve_frame_path(
            frame["frame_path"],
            FRAME_ROOT,
            dataset_name,
            group["video_name"],
        )
        original_paths.append(source)

    model_paths = list(original_paths)
    target_path = original_paths[15]

    return original_paths, model_paths, target_path


def build_context_text(
    group: Dict[str, Any],
    original_paths: List[Path],
) -> str:
    """
    Same annotation-free user text as the latest zero-shot script.

    `original_paths` is retained in the signature so the training and
    inference builders stay structurally parallel.
    """
    del original_paths

    target_frame = group["frames"][15]

    lines: List[str] = []

    lines.append(
        f"Video: {group['video_name']}"
    )

    lines.append(
        "The supplied video contains 16 raw frames in chronological order."
    )

    lines.append(
        "Frames 1-15 are visual temporal context only."
    )

    lines.append(
        "No person IDs, bounding boxes, tracking annotations, or "
        "interaction-readiness labels are provided for the context frames."
    )

    lines.append(
        "Frame 16 is the target frame. Detect and classify people in "
        "frame 16 using the target image and the preceding raw visual context."
    )

    lines.append(
        f"Target dataset frame number: "
        f"{int(target_frame['frame_number'])}"
    )

    lines.append(
        "Return target-frame bounding boxes in normalized [0,1000] xyxy coordinates."
    )

    return "\n".join(lines)


def normalize_label(label: Any) -> str:
    value = str(label).strip().lower()

    if value == "interaction_done":
        value = "not_interaction_ready"

    if value not in CLASS_NAMES:
        raise ValueError(
            f"Unsupported label: {label!r}"
        )

    return value


def build_target_json(
    group: Dict[str, Any],
    target_path: Path,
) -> str:
    """
    Build the supervised assistant response for TARGET frame 16 ONLY.

    Important:
    - Frame-16 boxes are converted from pixel-space GT to normalized [0,1000].
    - Output person_ID values are reassigned consecutively as 1..N to match
      the current zero-shot prompt.
    - Original dataset tracking IDs are NOT exposed to the model.
    """
    target_frame = group["frames"][15]

    width, height = get_image_size(
        target_path
    )

    target_people = list(
        target_frame.get("people", [])
    )

    # The temporal dataset should contain at most five selected people.
    if len(target_people) > 5:
        raise ValueError(
            f"Target frame contains {len(target_people)} ground-truth people. "
            "The current experiment expects at most 5 selected people per frame."
        )

    # Deterministic target ordering.
    # Dataset person_ID is used ONLY to make the target ordering stable.
    # It is not copied into the assistant output.
    target_people = sorted(
        target_people,
        key=lambda item: int(item["person_ID"]),
    )

    people_output: List[Dict[str, Any]] = []

    for output_id, person in enumerate(
        target_people,
        start=1,
    ):
        bbox = normalize_box_1000(
            person["bbox"],
            width,
            height,
        )

        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid zero-area target bbox in frame "
                f"{target_frame['frame_number']}: {bbox}"
            )

        people_output.append(
            {
                "person_ID": output_id,
                "bbox": bbox,
                "label": normalize_label(
                    person["label"]
                ),
            }
        )

    return json.dumps(
        {"people": people_output},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_prompt_messages(
    group: Dict[str, Any],
    original_paths: List[Path],
    model_paths: List[Path],
) -> List[Dict[str, Any]]:
    """
    Prompt seen by the fine-tuned model.

    This intentionally matches zero-shot inference:
        system prompt
        + 16 raw frames
        + annotation-free user instructions
    """
    context_text = build_context_text(
        group,
        original_paths,
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


def build_training_messages(
    group: Dict[str, Any],
    original_paths: List[Path],
    model_paths: List[Path],
    target_path: Path,
) -> List[Dict[str, Any]]:
    """
    Same input as zero-shot, with frame-16 GT appended ONLY as the
    assistant response used for supervised loss.
    """
    messages = build_prompt_messages(
        group,
        original_paths,
        model_paths,
    )

    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": build_target_json(
                        group,
                        target_path,
                    ),
                }
            ],
        }
    )

    return messages


def process_messages(
    processor,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool,
) -> Dict[str, torch.Tensor]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
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
        videos, video_metadatas = zip(
            *video_inputs
        )
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
        padding=False,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )

    return inputs


class TemporalJsonDataset(Dataset):
    def __init__(
        self,
        json_dir: Path,
        split: str,
        limit: Optional[int] = None,
    ):
        self.items: List[
            Tuple[Path, Dict[str, Any]]
        ] = []

        for json_path in sorted(
            json_dir.rglob("*.json")
        ):
            try:
                with json_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    group = json.load(file)
            except Exception as error:
                print(
                    f"[SKIP] {json_path.name}: {error}"
                )
                continue

            required = {
                "video_name",
                "frame_group",
                "split",
                "frames",
            }

            if not required.issubset(
                group.keys()
            ):
                continue

            if (
                str(group["split"])
                .strip()
                .lower()
                != split.lower()
            ):
                continue

            if len(group["frames"]) != 16:
                print(
                    f"[SKIP] {json_path.name}: "
                    f"expected 16 frames, "
                    f"got {len(group['frames'])}"
                )
                continue

            target_people = (
                group["frames"][15]
                .get("people", [])
            )

            if len(target_people) > 5:
                print(
                    f"[SKIP] {json_path.name}: "
                    f"target frame has {len(target_people)} people; "
                    "expected at most 5."
                )
                continue

            self.items.append(
                (json_path, group)
            )

        if limit is not None:
            self.items = self.items[:limit]

        print(
            f"[DATA] {split}: "
            f"{len(self.items)} temporal groups"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        json_path, group = self.items[
            index
        ]

        return {
            "json_path": str(json_path),
            "group": group,
        }


class TemporalSFTCollator:
    """
    Multimodal batch-size-1 SFT collator.

    Input:
        system prompt
        + 16 raw frames
        + annotation-free user text

    Target:
        frame-16 people + normalized bboxes + labels only

    Prompt tokens are masked with -100.
    Loss is computed only over assistant target tokens.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError(
                "This collator requires "
                "PER_DEVICE_*_BATCH_SIZE = 1."
            )

        feature = features[0]

        json_path = Path(
            feature["json_path"]
        )
        group = feature["group"]

        (
            original_paths,
            model_paths,
            target_path,
        ) = prepare_group_images(
            group,
            json_path,
        )

        prompt_messages = (
            build_prompt_messages(
                group,
                original_paths,
                model_paths,
            )
        )

        full_messages = (
            build_training_messages(
                group,
                original_paths,
                model_paths,
                target_path,
            )
        )

        # Prompt-only version ends with the assistant generation marker.
        # This length is masked from loss.
        prompt_inputs = process_messages(
            self.processor,
            prompt_messages,
            add_generation_prompt=True,
        )

        # Full SFT conversation includes the ground-truth assistant response.
        full_inputs = process_messages(
            self.processor,
            full_messages,
            add_generation_prompt=False,
        )

        labels = (
            full_inputs["input_ids"]
            .clone()
        )

        prompt_length = (
            prompt_inputs["input_ids"]
            .shape[1]
        )

        if prompt_length >= labels.shape[1]:
            raise RuntimeError(
                "Prompt length is not smaller than "
                "the full training sequence. "
                "Assistant-target masking cannot be applied safely."
            )

        labels[:, :prompt_length] = -100

        pad_token_id = (
            self.processor
            .tokenizer
            .pad_token_id
        )

        if pad_token_id is not None:
            labels[
                full_inputs["input_ids"]
                == pad_token_id
            ] = -100

        full_inputs["labels"] = labels

        return dict(full_inputs)


def find_last_checkpoint(
    output_dir: Path,
) -> Optional[str]:
    if not output_dir.exists():
        return None

    checkpoints: List[
        Tuple[int, Path]
    ] = []

    for path in output_dir.glob(
        "checkpoint-*"
    ):
        if not path.is_dir():
            continue

        try:
            step = int(
                path.name.split("-")[-1]
            )
        except ValueError:
            continue

        checkpoints.append(
            (step, path)
        )

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda item: item[0]
    )

    return str(
        checkpoints[-1][1]
    )


def load_model(
    dtype: torch.dtype,
):
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }

    if ATTN_IMPLEMENTATION != "auto":
        kwargs[
            "attn_implementation"
        ] = ATTN_IMPLEMENTATION

    print(
        f"[MODEL] Loading {MODEL_NAME}"
    )

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            MODEL_NAME,
            **kwargs,
        )
    )

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_NAME,
        )
    )

    if (
        processor.tokenizer.pad_token_id
        is None
    ):
        processor.tokenizer.pad_token = (
            processor.tokenizer.eos_token
        )

    # Optional visual-resolution limits.
    if MAX_PIXELS is not None:
        if hasattr(
            processor.image_processor,
            "max_pixels",
        ):
            processor.image_processor.max_pixels = (
                MAX_PIXELS
            )

    if MIN_PIXELS is not None:
        if hasattr(
            processor.image_processor,
            "min_pixels",
        ):
            processor.image_processor.min_pixels = (
                MIN_PIXELS
            )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model, processor


def make_training_arguments(
    dtype: torch.dtype,
    has_val: bool,
) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": str(
            OUTPUT_DIR
        ),
        "num_train_epochs": (
            NUM_TRAIN_EPOCHS
        ),
        "per_device_train_batch_size": (
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),
        "per_device_eval_batch_size": (
            PER_DEVICE_EVAL_BATCH_SIZE
        ),
        "gradient_accumulation_steps": (
            GRADIENT_ACCUMULATION_STEPS
        ),
        "learning_rate": (
            LEARNING_RATE
        ),
        "weight_decay": (
            WEIGHT_DECAY
        ),
        "warmup_steps": (
            WARMUP_STEPS
        ),
        "logging_steps": (
            LOGGING_STEPS
        ),
        "save_steps": (
            SAVE_STEPS
        ),
        "save_total_limit": (
            SAVE_TOTAL_LIMIT
        ),
        "remove_unused_columns": False,
        "report_to": "none",
        "dataloader_num_workers": (
            DATALOADER_NUM_WORKERS
        ),
        "bf16": (
            dtype == torch.bfloat16
        ),
        "fp16": (
            dtype == torch.float16
        ),
        "optim": "adamw_torch",
        "lr_scheduler_type": "cosine",
        "gradient_checkpointing": (
            GRADIENT_CHECKPOINTING
        ),
    }

    # Compatibility across Transformers versions.
    parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if has_val:
        if "eval_strategy" in parameters:
            kwargs["eval_strategy"] = (
                "steps"
            )
        elif (
            "evaluation_strategy"
            in parameters
        ):
            kwargs[
                "evaluation_strategy"
            ] = "steps"

        if "eval_steps" in parameters:
            kwargs["eval_steps"] = (
                EVAL_STEPS
            )

    return TrainingArguments(
        **kwargs
    )


class CudaCleanupCallback(
    TrainerCallback
):
    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        gc.collect()
        torch.cuda.empty_cache()
        return control


def main() -> None:
    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dtype = check_cuda()

    train_dataset = (
        TemporalJsonDataset(
            json_dir=JSON_DIR,
            split="Train",
            limit=MAX_TRAIN_GROUPS,
        )
    )

    val_dataset = (
        TemporalJsonDataset(
            json_dir=JSON_DIR,
            split="Val",
            limit=MAX_VAL_GROUPS,
        )
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "No Train temporal groups were found."
        )

    print(
        "[DATA] Train = gradient updates; "
        "Val = validation only; "
        "Test = untouched."
    )

    print(
        "[TASK] Input = 16 RAW frames only. "
        "No context annotations are exposed."
    )

    print(
        "[TASK] Supervised target = frame-16 "
        "people/bboxes/labels only."
    )

    model, processor = load_model(
        dtype
    )

    collator = TemporalSFTCollator(
        processor
    )

    training_args = (
        make_training_arguments(
            dtype=dtype,
            has_val=(
                len(val_dataset) > 0
            ),
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=(
            val_dataset
            if len(val_dataset) > 0
            else None
        ),
        data_collator=collator,
        callbacks=[
            CudaCleanupCallback()
        ],
    )

    resume_checkpoint = None

    if RESUME_FROM_LAST_CHECKPOINT:
        resume_checkpoint = (
            find_last_checkpoint(
                OUTPUT_DIR
            )
        )

        if resume_checkpoint:
            print(
                "[TRAIN] Resuming from "
                f"{resume_checkpoint}"
            )
        else:
            print(
                "[TRAIN] No checkpoint found; "
                "starting from base model + "
                "new LoRA adapter."
            )

    trainer.train(
        resume_from_checkpoint=(
            resume_checkpoint
        )
    )

    final_adapter_dir = (
        OUTPUT_DIR
        / "final_adapter"
    )

    print(
        "[SAVE] Saving LoRA adapter to "
        f"{final_adapter_dir}"
    )

    trainer.model.save_pretrained(
        final_adapter_dir
    )

    processor.save_pretrained(
        final_adapter_dir
    )

    print("[DONE]")
    print(
        f"LoRA adapter: "
        f"{final_adapter_dir}"
    )


if __name__ == "__main__":
    main()
