#!/usr/bin/env python3
"""
Zero-shot temporal detection + interaction-readiness inference with Qwen3-VL.

CORRECT TEMPORAL TASK
---------------------
Each input JSON is one 16-frame sliding window.

Frames 1-15:
    - raw images are shown to the model as temporal context
    - NO person IDs, bounding boxes, or interaction-readiness labels are supplied

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
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ==============================================================================
# USER CONFIGURATION — EDIT THESE VALUES
# ==============================================================================

JSON_DIR = Path("/home/marilyn/Downloads/data/JSON_temporal")

# Set to None if frame_path values inside the JSON already work on this machine.
# Otherwise point this to the root of your frames directory.
FRAME_ROOT = Path("/home/marilyn/Downloads/data/frames")

OUTPUT_DIR = Path("/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/videoLevel_zeroShot/Qwen3VL-8B")

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# "Train", "Val", "Test", or "All"
SPLIT = "Test"

MAX_NEW_TOKENS = 1024

# 0.0 = greedy/deterministic
TEMPERATURE = 0.0

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


def prepare_group_images(
    group: Dict[str, Any],
    json_path: Path,
    temp_dir: Path,
) -> Tuple[List[Path], List[Path], Path]:
    """
    Returns:
        original_paths:
            all 16 original frame paths

        model_paths:
            the same 16 raw, unannotated frame paths given to Qwen

        target_path:
            raw frame-16 path

    No visual annotations are added to context frames.
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

    # All 16 frames are passed to Qwen exactly as they are on disk.
    model_paths = list(original_paths)
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
    Build annotation-free temporal instructions.

    Frames 1-15 contribute ONLY their raw visual content.
    No person IDs, bounding boxes, or readiness labels from those frames are
    exposed to the model.

    Frame-16 ground truth is also excluded and is used only after inference.
    """
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
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse each predicted person independently so one malformed entry does not erase the whole frame."""
    raw_people = parsed.get("people")

    if not isinstance(raw_people, list):
        raise ValueError(
            "Model JSON must contain a list field named 'people'."
        )

    predictions: List[Dict[str, Any]] = []
    person_parse_errors: List[str] = []

    for index, person in enumerate(raw_people):
        try:
            if not isinstance(person, dict):
                raise ValueError(f"people[{index}] is not a JSON object.")

            if "bbox" not in person:
                raise ValueError(f"people[{index}] is missing bbox.")

            if "label" not in person:
                raise ValueError(f"people[{index}] is missing label.")

            bbox = person["bbox"]
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"people[{index}].bbox must have four coordinates.")

            pixel_bbox = denormalize_box_1000(
                [float(value) for value in bbox],
                target_width,
                target_height,
            )

            x1, y1, x2, y2 = pixel_bbox
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"people[{index}] produced a zero-area bbox.")

            raw_person_id = person.get("person_ID", person.get("person_id"))
            if raw_person_id is None:
                person_id = index + 1
            else:
                try:
                    person_id = int(raw_person_id)
                except (TypeError, ValueError):
                    person_id = index + 1

            predictions.append({
                "person_id": person_id,
                "bbox_2d": pixel_bbox,
                "label": normalize_predicted_label(person["label"]),
                "bbox_normalized_1000": [float(value) for value in bbox],
            })

        except Exception as error:
            person_parse_errors.append(f"{type(error).__name__}: {error}")

    return predictions, person_parse_errors


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
    person_parse_errors: List[str] = []
    parsed_model_response = None

    try:
        parsed_model_response = extract_json_object(
            raw_response
        )

        predicted_people, person_parse_errors = parse_predicted_people(
            parsed_model_response,
            target_width,
            target_height,
        )

        if person_parse_errors:
            parse_error = " | ".join(person_parse_errors)

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
        "parsed_model_response": (
            parsed_model_response
        ),
        "person_parse_errors": (
            person_parse_errors
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

        # Save the decoded model output BEFORE any evaluation/parsing logic can hide it.
        raw_response_path = (
            OUTPUT_DIR
            / (json_path.stem + "__raw_response.txt")
        )

        with raw_response_path.open(
            "w",
            encoding="utf-8",
        ) as raw_file:
            raw_file.write(
                str(record.get("raw_model_response", ""))
            )

        parsed_response = record.get("parsed_model_response")
        if parsed_response is not None:
            parsed_response_path = (
                OUTPUT_DIR
                / (json_path.stem + "__parsed_response.json")
            )
            with parsed_response_path.open(
                "w",
                encoding="utf-8",
            ) as parsed_file:
                json.dump(
                    parsed_response,
                    parsed_file,
                    indent=2,
                    ensure_ascii=False,
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
