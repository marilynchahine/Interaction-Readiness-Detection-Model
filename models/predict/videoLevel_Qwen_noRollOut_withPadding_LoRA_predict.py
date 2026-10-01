#!/usr/bin/env python3
"""
Zero-shot temporal detection + interaction-readiness inference with Qwen3-VL.

CORRECT TEMPORAL TASK
---------------------
Each input JSON is one 16-frame sliding window.

Frames 1-15:
    - image is shown to the model
    - person_ID, bbox, and interaction-readiness label are supplied as temporal context

Frame 16:
    - image is shown to the model
    - NO person annotations from frame 16 are shown to the model
    - the model must:
        1. detect every visible person
        2. predict a bounding box for each detected person
        3. classify each detected person as:
              not_interaction_ready
              interaction_ready
              interaction_ongoing

The frame-16 ground truth is used ONLY after inference for evaluation output.

OUTPUT
------
The combined predictions.jsonl is deliberately written in the schema expected
by the existing Qwen evaluation script:

{
    "source_dataset": "...",
    "video_id": "...",
    "frame_id": ...,
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

Predicted boxes are requested from Qwen in normalized [0,1000] xyxy coordinates
relative to TARGET frame 16, then converted back to original pixel coordinates
before being saved as bbox_2d. This makes them directly comparable with the
ground-truth pixel-space boxes in the evaluator.

GPU-only.

Suggested packages:
    pip install -U torch torchvision transformers accelerate qwen-vl-utils pillow tqdm
"""

import gc
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from peft import PeftModel


# ==============================================================================
# USER CONFIGURATION — EDIT THESE VALUES
# ==============================================================================

JSON_DIR = Path("/home/marilyn/Downloads/data/JSON_temporal")

# Set to None if frame_path values inside the JSON already work on this machine.
# Otherwise point this to the root of your frames directory.
FRAME_ROOT = Path("/home/marilyn/Downloads/data/frames")

OUTPUT_DIR = Path("/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/videoLevel_fineTuned/Qwen3VL_2B_Instruct/inference_noRollOut")

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
LORA_CHECKPOINT = Path("/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/videoLevel_fineTuned/Qwen3VL_2B_Instruct/final_adapter")

# "Train", "Val", "Test", or "All"
SPLIT = "Test"

MAX_NEW_TOKENS = 1024

# 0.0 = greedy/deterministic
TEMPERATURE = 0.0

# Draw bbox + person ID on temporary copies of CONTEXT frames 1-15.
# The target frame 16 is NEVER annotated.
ANNOTATE_CONTEXT_FRAMES = True

# Skip already completed per-group prediction files.
RESUME = True

# Set e.g. 10 for a quick test, or None for all groups.
LIMIT = None

# "auto", "flash_attention_2", "sdpa", or "eager"
ATTN_IMPLEMENTATION = "auto"

# ==============================================================================


CLASS_NAMES = {
    "not_interaction_ready",
    "interaction_ready",
    "interaction_ongoing",
}


SYSTEM_PROMPT = """You are an expert in human-robot interaction, temporal human behavior analysis, and visual grounding.

Your task is temporal person detection and interaction-readiness classification.

You will receive a short ordered sequence of 16 video frames.

TEMPORAL STRUCTURE

- Frames 1 through 15 are CONTEXT frames.
- For each context frame, annotation metadata is provided for the tracked people visible in that frame.
- Frame 16 is the TARGET frame.
- No person annotations from frame 16 are provided to you.
- You must infer the people present in frame 16 from the target image itself, while using frames 1-15 as temporal context.

CONTEXT ANNOTATION STRUCTURE

For each of frames 1-15, the annotation metadata contains:

- frame_number:
  The original frame number in the source video.

- people:
  The tracked people visible in that context frame.

For every annotated person:

- person_ID:
  A persistent tracked identity across the context sequence.
  The same person_ID refers to the same individual across frames whenever that person remains visible.

- bbox:
  The person's bounding box in that context frame.
  Format: [x1, y1, x2, y2].
  The textual bbox coordinates are normalized to the range 0-1000 relative to that frame.

- label:
  The person's ground-truth interaction-readiness state in that CONTEXT frame.

The context labels are intentionally supplied so you can reason about how each person's interaction state evolves over time.

TARGET FRAME TASK

For frame 16, you must independently inspect the target image and:

1. Detect every visible person that should be considered for interaction-readiness prediction.
2. Predict one bounding box for each detected person.
3. Predict one interaction-readiness label for each detected person.

Use the previous 15 frames to reason about temporal continuity, motion, approach/withdrawal, body orientation, gaze/head direction when visible, stopping, gestures, posture changes, and whether an interaction is beginning or already occurring.

VALID LABELS

- not_interaction_ready:
  The person is not currently signaling readiness or intention to initiate an interaction.

- interaction_ready:
  The person is signaling readiness or intention to initiate an interaction,
  but the interaction itself has not yet started.

- interaction_ongoing:
  The person is already actively engaged in the interaction.

BOUNDING-BOX OUTPUT

For TARGET frame 16:
- bbox must use [x1, y1, x2, y2].
- Coordinates must be normalized to 0-1000 relative to the target frame.
- 0 is the left/top edge.
- 1000 is the right/bottom edge.
- Ensure x2 > x1 and y2 > y1.

PERSON ID OUTPUT

If a detected target-frame person can clearly be associated with one of the tracked person_ID values from frames 1-15, preserve that person_ID.
If a target-frame person cannot be confidently associated with a previous track, use null for person_ID.
Person identity is supplementary; detection evaluation is based on bounding-box matching.

IMPORTANT RULES

- Frames 1-15 are context.
- Frame 16 is the only frame you predict.
- Never output predictions for frames 1-15.
- Do not assume that every context person remains visible in frame 16.
- Do not assume that no new person can appear in frame 16.
- Detect people from the actual target image.
- Do not copy a context-frame bbox as the target bbox without visually localizing the person in frame 16.
- Return each target-frame person exactly once.
- Return ONLY valid JSON.
- Do not include markdown, explanation, prose, or comments.

Required output schema:

{"people":[
  {
    "person_ID":1,
    "bbox":[x1,y1,x2,y2],
    "label":"interaction_ready"
  }
]}

If no person is visible in the target frame, return:

{"people":[]}"""


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
        
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        **kwargs,
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_CHECKPOINT,
    )

    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(f"[MODEL] loaded on {model.device}")

    return model, processor


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


def denormalize_box_1000(
    box: List[float],
    width: int,
    height: int,
) -> List[float]:
    """
    Convert Qwen's requested target-frame [0,1000] box back to
    original image pixel coordinates.
    """
    x1, y1, x2, y2 = map(float, box)

    # Clamp normalized coordinates before conversion.
    x1 = max(0.0, min(x1, 1000.0))
    x2 = max(0.0, min(x2, 1000.0))
    y1 = max(0.0, min(y1, 1000.0))
    y2 = max(0.0, min(y2, 1000.0))

    # Correct reversed model outputs conservatively.
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    return [
        left * width / 1000.0,
        top * height / 1000.0,
        right * width / 1000.0,
        bottom * height / 1000.0,
    ]


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def annotate_context_frame(
    source_path: Path,
    people: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Draw bbox + ID on context frames only.

    The readiness labels are supplied textually rather than drawn.
    The target frame is never passed through this function.
    """
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        width, height = image.size

        draw = ImageDraw.Draw(image)

        line_width = max(
            2,
            round(min(width, height) / 180),
        )
        text_pad = max(2, line_width)

        for person in sorted(
            people,
            key=lambda item: int(item["person_ID"]),
        ):
            person_id = int(person["person_ID"])

            x1, y1, x2, y2 = clamp_box(
                person["bbox"],
                width,
                height,
            )

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=line_width,
            )

            text = f"ID {person_id}"

            try:
                text_box = draw.textbbox(
                    (x1, y1),
                    text,
                )
                text_width = (
                    text_box[2] - text_box[0]
                )
                text_height = (
                    text_box[3] - text_box[1]
                )
            except Exception:
                text_width = 40
                text_height = 12

            label_top = max(
                0,
                y1 - text_height - 2 * text_pad,
            )

            draw.rectangle(
                [
                    x1,
                    label_top,
                    min(
                        width,
                        x1
                        + text_width
                        + 2 * text_pad,
                    ),
                    y1,
                ],
                fill="red",
            )

            draw.text(
                (
                    x1 + text_pad,
                    label_top + text_pad,
                ),
                text,
                fill="white",
            )

        image.save(
            output_path,
            quality=95,
        )


def prepare_group_images(
    group: Dict[str, Any],
    json_path: Path,
    temp_dir: Path,
) -> Tuple[List[Path], List[Path], Path]:
    """
    Returns:
        original_paths:
            all 16 original paths

        model_paths:
            all 16 paths given to Qwen;
            frames 1-15 may be annotated copies,
            frame 16 is ALWAYS the raw target image

        target_path:
            raw frame-16 path
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
    model_paths: List[Path] = []

    for index, frame in enumerate(group["frames"]):
        source = resolve_frame_path(
            frame["frame_path"],
            FRAME_ROOT,
            dataset_name,
            group["video_name"],
        )

        original_paths.append(source)

        # First 15 = context.
        if index < 15 and ANNOTATE_CONTEXT_FRAMES:
            temporary = (
                temp_dir
                / f"context_{index + 1:02d}_{source.name}"
            )

            annotate_context_frame(
                source_path=source,
                people=frame.get("people", []),
                output_path=temporary,
            )

            model_paths.append(temporary)

        else:
            # Critical: target frame 16 remains completely unannotated.
            model_paths.append(source)

    target_path = original_paths[15]

    return (
        original_paths,
        model_paths,
        target_path,
    )


def build_context_text(
    group: Dict[str, Any],
    original_paths: List[Path],
) -> str:
    """
    Include frame 1-15 annotations INCLUDING context readiness labels.
    Exclude ALL annotations from frame 16.
    """
    context_frames = group["frames"][:15]
    target_frame = group["frames"][15]

    lines: List[str] = []

    lines.append(
        f"Video: {group['video_name']}"
    )

    lines.append(
        f"Frames 1-15 are annotated temporal context. "
        f"Frame 16 is the unannotated target."
    )

    lines.append(
        f"Target dataset frame number: "
        f"{int(target_frame['frame_number'])}"
    )

    lines.append(
        "The supplied video contains all 16 images in order. "
        "The textual annotations below correspond ONLY to video frames 1-15."
    )

    lines.append(
        "Context bounding boxes use normalized xyxy coordinates "
        "in the range 0-1000 relative to each context frame."
    )

    lines.append("")
    lines.append("CONTEXT ANNOTATIONS:")

    for temporal_index, (
        frame,
        frame_path,
    ) in enumerate(
        zip(
            context_frames,
            original_paths[:15],
        ),
        start=1,
    ):
        width, height = get_image_size(
            frame_path
        )

        frame_number = int(
            frame["frame_number"]
        )

        people = sorted(
            frame.get("people", []),
            key=lambda item: int(
                item["person_ID"]
            ),
        )

        if not people:
            lines.append(
                f"- temporal_frame_index={temporal_index}, "
                f"dataset_frame_number={frame_number}: "
                f"no annotated people"
            )
            continue

        entries: List[str] = []

        for person in people:
            person_id = int(
                person["person_ID"]
            )

            bbox = normalize_box_1000(
                person["bbox"],
                width,
                height,
            )

            label = str(
                person["label"]
            ).strip()

            entries.append(
                f"ID {person_id}: "
                f"bbox={bbox}, "
                f"label={label}"
            )

        lines.append(
            f"- temporal_frame_index={temporal_index}, "
            f"dataset_frame_number={frame_number}: "
            + "; ".join(entries)
        )

    lines.append("")
    lines.append(
        "TARGET FRAME 16:"
    )
    lines.append(
        f"- temporal_frame_index=16"
    )
    lines.append(
        f"- dataset_frame_number="
        f"{int(target_frame['frame_number'])}"
    )
    lines.append(
        "- No target-frame person annotations are provided."
    )
    lines.append(
        "- Detect and classify the people directly from the target image."
    )
    lines.append(
        "- Return target-frame bbox coordinates normalized to 0-1000."
    )

    return "\n".join(lines)


def build_messages(
    group: Dict[str, Any],
    original_paths: List[Path],
    model_paths: List[Path],
) -> List[Dict[str, Any]]:
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
                    # IMPORTANT on Windows:
                    # use ordinary absolute paths rather than .as_uri().
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


def extract_json_object(
    text: str,
) -> Dict[str, Any]:
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

    if (
        start != -1
        and end != -1
        and end > start
    ):
        candidate = cleaned[
            start : end + 1
        ]

        parsed = json.loads(candidate)

        if isinstance(parsed, dict):
            return parsed

    raise ValueError(
        "Model response is not valid JSON."
    )


def normalize_predicted_label(
    value: Any,
) -> str:
    label = str(value).strip().lower()

    if label == "interaction_done":
        label = "not_interaction_ready"

    if label not in CLASS_NAMES:
        raise ValueError(
            f"Unsupported predicted label: {value!r}"
        )

    return label


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

    for index, person in enumerate(
        raw_people
    ):
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

        pixel_bbox = denormalize_box_1000(
            [float(value) for value in bbox],
            target_width,
            target_height,
        )

        # Prevent zero-area boxes from entering the evaluator.
        x1, y1, x2, y2 = pixel_bbox

        if x2 <= x1 or y2 <= y1:
            continue

        raw_person_id = person.get(
            "person_ID",
            person.get("person_id"),
        )

        if raw_person_id is None:
            person_id = None
        else:
            try:
                person_id = int(raw_person_id)
            except (TypeError, ValueError):
                person_id = None

        predictions.append(
            {
                "person_id": person_id,
                "bbox_2d": pixel_bbox,
                "label": normalize_predicted_label(
                    person["label"]
                ),
                # Save normalized model output too for debugging.
                "bbox_normalized_1000": [
                    float(value)
                    for value in bbox
                ],
            }
        )

    return predictions


def normalize_gt_label(
    value: Any,
) -> str:
    label = str(value).strip().lower()

    if label == "interaction_done":
        return "not_interaction_ready"

    return label


def build_ground_truth_people(
    target_frame: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ground_truth: List[Dict[str, Any]] = []

    for person in target_frame.get(
        "people",
        [],
    ):
        original_label = str(
            person["label"]
        ).strip()

        ground_truth.append(
            {
                "person_id": int(
                    person["person_ID"]
                ),
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


@torch.inference_mode()
def run_group(
    model,
    processor,
    group: Dict[str, Any],
    json_path: Path,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="qwen_temporal_"
    ) as temp_directory:
        temp_dir = Path(temp_directory)

        (
            original_paths,
            model_paths,
            target_path,
        ) = prepare_group_images(
            group,
            json_path,
            temp_dir,
        )

        target_width, target_height = (
            get_image_size(target_path)
        )

        messages = build_messages(
            group,
            original_paths,
            model_paths,
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
            (
                videos,
                video_metadatas,
            ) = zip(*video_inputs)

            videos = list(videos)
            video_metadatas = list(
                video_metadatas
            )
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

        generation_kwargs: Dict[
            str,
            Any,
        ] = {
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
            generation_kwargs[
                "do_sample"
            ] = False

        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

        generated_trimmed = [
            output_ids[
                len(input_ids) :
            ]
            for input_ids, output_ids
            in zip(
                inputs.input_ids,
                generated_ids,
            )
        ]

        raw_response = (
            processor.batch_decode(
                generated_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

    parse_error = None

    try:
        parsed = extract_json_object(
            raw_response
        )

        predicted_people = (
            parse_predicted_people(
                parsed,
                target_width,
                target_height,
            )
        )

    except Exception as error:
        predicted_people = []
        parse_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    target_frame = group["frames"][15]

    source_dataset = (
        dataset_name_from_json_file(
            json_path,
            group["video_name"],
        )
    )

    record = {
        # Fields required by your existing evaluator:
        "source_dataset": source_dataset,
        "video_id": str(
            group["video_name"]
        ),
        "frame_id": int(
            target_frame["frame_number"]
        ),
        "ground_truth_people": (
            build_ground_truth_people(
                target_frame
            )
        ),
        "predicted_people": predicted_people,

        # Error fields expected/used by evaluator:
        "image_error": None,
        "parse_error": parse_error,

        # Extra temporal metadata:
        "frame_group": int(
            group["frame_group"]
        ),
        "split": str(
            group["split"]
        ),
        "source_json": json_path.name,
        "model": MODEL_NAME,
        "zero_shot": True,
        "context_frame_numbers": [
            int(frame["frame_number"])
            for frame in group["frames"][:15]
        ],
        "target_frame_number": int(
            target_frame["frame_number"]
        ),
        "target_image_width": (
            target_width
        ),
        "target_image_height": (
            target_height
        ),
        "raw_model_response": (
            raw_response
        ),
    }

    return record


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


def load_groups() -> List[
    Tuple[Path, Dict[str, Any]]
]:
    selected: List[
        Tuple[Path, Dict[str, Any]]
    ] = []

    json_files = sorted(
        JSON_DIR.rglob("*.json")
    )

    for json_path in json_files:
        try:
            with json_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                group = json.load(file)
        except Exception as error:
            print(
                f"[SKIP] {json_path.name}: "
                f"{error}"
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

        if not split_matches(
            str(group["split"]),
            SPLIT,
        ):
            continue

        if len(group["frames"]) != 16:
            print(
                f"[SKIP] {json_path.name}: "
                f"expected 16 frames, "
                f"got {len(group['frames'])}"
            )
            continue

        selected.append(
            (json_path, group)
        )

    if LIMIT is not None:
        selected = selected[:LIMIT]

    print(
        f"[DATA] JSON files found: "
        f"{len(json_files)}"
    )
    print(
        f"[DATA] Selected "
        f"{len(selected)} groups "
        f"for split={SPLIT}"
    )

    return selected


def rebuild_combined_jsonl(
    output_dir: Path,
    combined_path: Path,
) -> None:
    """
    Rebuild predictions.jsonl from completed per-group output JSON files.

    This avoids duplicate JSONL rows when RESUME=True and the script is restarted.
    """
    prediction_files = sorted(
        output_dir.glob(
            "*__predictions.json"
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


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dtype = check_cuda()

    model, processor = load_model(
        dtype
    )

    selected = load_groups()

    for json_path, group in tqdm(
        selected,
        desc="Temporal inference",
    ):
        output_path = (
            OUTPUT_DIR
            / (
                json_path.stem
                + "__predictions.json"
            )
        )

        if (
            RESUME
            and output_path.exists()
        ):
            continue

        try:
            record = run_group(
                model=model,
                processor=processor,
                group=group,
                json_path=json_path,
            )

        except Exception as error:
            # Keep a record in evaluator-compatible form.
            target_frame = (
                group["frames"][15]
            )

            record = {
                "source_dataset": (
                    dataset_name_from_json_file(
                        json_path,
                        group["video_name"],
                    )
                ),
                "video_id": str(
                    group["video_name"]
                ),
                "frame_id": int(
                    target_frame[
                        "frame_number"
                    ]
                ),
                "ground_truth_people": (
                    build_ground_truth_people(
                        target_frame
                    )
                ),
                "predicted_people": [],
                "image_error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
                "parse_error": None,
                "frame_group": int(
                    group["frame_group"]
                ),
                "split": str(
                    group["split"]
                ),
                "source_json": (
                    json_path.name
                ),
                "model": MODEL_NAME,
                "zero_shot": True,
            }

            print(
                f"\n[ERROR] "
                f"{json_path.name}: "
                f"{error}"
            )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                record,
                file,
                indent=2,
                ensure_ascii=False,
            )

        if record.get(
            "parse_error"
        ):
            print(
                f"\n[PARSE ERROR] "
                f"{json_path.name}: "
                f"{record['parse_error']}"
            )

        gc.collect()
        torch.cuda.empty_cache()

    combined_path = (
        OUTPUT_DIR
        / "predictions.jsonl"
    )

    rebuild_combined_jsonl(
        OUTPUT_DIR,
        combined_path,
    )

    print("\n[DONE]")
    print(
        f"Per-group outputs: "
        f"{OUTPUT_DIR}"
    )
    print(
        f"Combined evaluator JSONL: "
        f"{combined_path}"
    )


if __name__ == "__main__":
    main()
