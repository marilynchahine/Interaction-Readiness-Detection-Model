from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

import matplotlib.pyplot as plt

from peft import PeftModel

# =============================================================================
# Configuration — edit these values directly
# =============================================================================

# Final balanced frame-based CSV dataset.
INPUT_CSV = Path(
    r"/home/marilyn/Downloads/data/balanced_frame_based_dataset_linux.csv"
)

# Inference will run only on rows belonging to this split.
# Matching is case-insensitive, so "test" matches CSV value "Test".
INFERENCE_SPLIT = "test"

# Folder where predictions, raw responses, and annotated images are saved.
OUTPUT_DIR = Path(r"/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/resized/Qwen3VL_8B_Instruct_LoRA/inference_results")

LORA_CHECKPOINT = Path(r"/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/resized/Qwen3VL_8B_Instruct_LoRA/final_adapter")

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
LOAD_IN_4BIT = False
MAX_NEW_TOKENS = 1024

# Final CSV columns.
SOURCE_DATASET_COLUMN = "source_dataset"
VIDEO_NAME_COLUMN = "video_name"
FRAME_NUMBER_COLUMN = "frame_number"
FRAME_LABEL_COLUMN = "frame_label"
PERSON_ID_COLUMN = "person_id"
PERSON_LABEL_COLUMN = "person_label"
BBOX_COLUMN = "bbox"
FRAME_PATH_COLUMN = "frame_path"
SPLIT_COLUMN = "split"

# Optional fallback base directory if frame_path values are relative.
# Your current CSV stores absolute Windows paths, so leave this as None.
IMAGE_BASE_DIRECTORY: Path | None = None

# One of: "original", "processed", "normalized_1000".
COORDINATE_MODE = "original"

SAVE_ANNOTATED = False
OVERWRITE = False

# Display the first N unique test frames before inference to verify image loading.
# Set to 0 to disable previews.
PREVIEW_FIRST_N = 0

# Set to an integer for a small test run, or None to process the whole split.
MAX_FRAMES: int | None = None

VALID_COORDINATE_MODES = {"original", "processed", "normalized_1000"}
VALID_LABELS = {
    "interaction_ready",
    "not_interaction_ready",
    "interaction_ongoing",
    "interaction_done",
}

InteractionLabel = Literal[
    "interaction_ready",
    "not_interaction_ready",
    "interaction_ongoing",
]


class PersonPrediction(TypedDict):
    person_id: int
    bbox: list[int]
    label: InteractionLabel


PROMPT = """
Analyze the provided image.

Your task is to:

1. Detect clearly visible human people in the image.
2. Return one bounding box for each detected person.
3. Assign exactly one interaction-readiness label to each person.
4. Detect AT MOST 5 people. If more than 5 people are visible, return only the 5 most prominent / clearly visible people. The "people" array MUST contain between 0 and 5 objects. Never return more than 5 people.

Use these label definitions:

- "interaction_ready":
  The person appears available, willing, or prepared to begin an interaction.
  This may include orienting toward the interaction partner, approaching,
  waiting for interaction, seeking attention, making eye contact, or otherwise
  showing readiness to engage. The interaction has not begun yet.

- "not_interaction_ready":
  The person does not appear ready to begin an interaction. This also includes
  people whose interaction is already finished. A person walking away, occupied
  with another activity, ignoring the interaction partner, oriented elsewhere,
  or showing no intention to engage belongs to this class.

- "interaction_ongoing":
  The person is already actively interacting with the interaction partner.
  The interaction has clearly started.

Important instructions:

- Classify every visible person separately.
- Base the label only on visible evidence in the image.
- Do not invent people who are not visible.
- Include a partially visible person when a meaningful box can be produced.
- Bounding boxes must use [x1, y1, x2, y2].
- Bounding-box coordinates must be normalized to the range 0 to 1000 relative to the full image.
- x coordinates are relative to the full image width.
- y coordinates are relative to the full image height.
- The top-left corner of the image is [0, 0].
- The bottom-right corner of the image is [1000, 1000].
- All bounding-box coordinates must be integers between 0 and 1000.
- Ensure x1 < x2 and y1 < y2.
- Return only valid JSON.
- Do not include Markdown fences.
- Do not include explanations outside the JSON.
- Use only:
  "interaction_ready",
  "not_interaction_ready",
  "interaction_ongoing".

Return in the format:

{
  "people": [
    {
      "person_id": 1,
      "bbox": [x1, y1, x2, y2],
      "label": "interaction_ready"
    },
    {
      "person_id": 2,
      "bbox": [x3, y3, x4, y4],
      "label": "not_interaction_ready"
    },
    {
      "person_id": 3,
      "bbox": [x5, y5, x6, y6],
      "label": "interaction_ongoing"
    }
  ]
}

When no person is visible, return:

{
  "people": []
}
""".strip()


# =============================================================================
# Dataset loading and validation
# =============================================================================

def load_csv_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the final balanced frame-based CSV dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    print(f"Loading: {csv_path}")
    dataset = pd.read_csv(csv_path)

    required_columns = {
        SOURCE_DATASET_COLUMN,
        VIDEO_NAME_COLUMN,
        FRAME_NUMBER_COLUMN,
        FRAME_LABEL_COLUMN,
        PERSON_ID_COLUMN,
        PERSON_LABEL_COLUMN,
        BBOX_COLUMN,
        FRAME_PATH_COLUMN,
        SPLIT_COLUMN,
    }
    missing = sorted(required_columns - set(dataset.columns))
    if missing:
        raise ValueError(
            "The CSV dataset is missing required columns: " + ", ".join(missing)
        )

    # Normalize string-valued identifier/label columns.
    for column in [
        SOURCE_DATASET_COLUMN,
        VIDEO_NAME_COLUMN,
        FRAME_LABEL_COLUMN,
        PERSON_ID_COLUMN,
        PERSON_LABEL_COLUMN,
        FRAME_PATH_COLUMN,
    ]:
        dataset[column] = dataset[column].astype(str).str.strip()

    # Keep split missingness visible instead of turning NaN into the string "nan".
    dataset[SPLIT_COLUMN] = dataset[SPLIT_COLUMN].astype("string").str.strip()

    invalid_frame_labels = sorted(
        set(dataset[FRAME_LABEL_COLUMN].dropna()) - VALID_LABELS
    )
    invalid_person_labels = sorted(
        set(dataset[PERSON_LABEL_COLUMN].dropna()) - VALID_LABELS
    )
    if invalid_frame_labels:
        raise ValueError(f"Unsupported frame labels found: {invalid_frame_labels}")
    if invalid_person_labels:
        raise ValueError(f"Unsupported person labels found: {invalid_person_labels}")

    # Within a unique frame, frame_path, frame_label and split must be identical
    # across all person rows.
    frame_key = [SOURCE_DATASET_COLUMN, VIDEO_NAME_COLUMN, FRAME_NUMBER_COLUMN]
    grouped = dataset.groupby(frame_key, dropna=False)

    for column in [FRAME_PATH_COLUMN, FRAME_LABEL_COLUMN, SPLIT_COLUMN]:
        inconsistent = grouped[column].nunique(dropna=False) > 1
        if inconsistent.any():
            bad_keys = inconsistent[inconsistent].index.tolist()[:10]
            raise ValueError(
                f"Column {column!r} is inconsistent within some frames. "
                f"Examples: {bad_keys}"
            )

    return dataset


def filter_dataset_to_split(
    dataset: pd.DataFrame,
    requested_split: str,
) -> pd.DataFrame:
    """Keep only rows assigned to the requested split in the same CSV."""
    requested_split = requested_split.strip().lower()
    if requested_split not in {"train", "val", "test"}:
        raise ValueError("INFERENCE_SPLIT must be train, val, or test.")

    split_normalized = dataset[SPLIT_COLUMN].str.lower()
    missing_split_rows = int(split_normalized.isna().sum())
    if missing_split_rows:
        print(
            f"Warning: {missing_split_rows:,} rows have no split value and will "
            "not be used for inference."
        )

    filtered = dataset.loc[split_normalized == requested_split].copy()
    filtered.reset_index(drop=True, inplace=True)

    if filtered.empty:
        available = sorted(
            dataset[SPLIT_COLUMN].dropna().astype(str).str.lower().unique().tolist()
        )
        raise ValueError(
            f"No rows were assigned to split {requested_split!r}. "
            f"Available split values: {available}"
        )

    return filtered


def normalize_scalar(value: Any) -> Any:
    """Convert NumPy/Pandas scalar values into normal Python values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def parse_bbox(value: Any) -> list[float]:
    """
    Parse the CSV bbox field.

    The final CSV stores bbox as a string such as:
        "[664.04, 1.55, 55.62, 105.71]"
    representing [x1, y1, x2, y2].
    """
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Could not parse bbox string: {value!r}") from exc

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Invalid ground-truth bbox: {value!r}")

    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Ground-truth bbox contains non-numeric values: {value!r}") from exc


def normalize_bbox(bbox: Any) -> list[int]:
    """Convert CSV [x, y, width, height] bbox to [x1, y1, x2, y2]."""
    x, y, width, height = parse_bbox(bbox)
    return [
        int(round(x)),
        int(round(y)),
        int(round(x + width)),
        int(round(y + height)),
    ]


def normalize_label(label: str) -> InteractionLabel:
    normalized = str(label).strip().lower()

    # Kept for compatibility if interaction_done ever appears again.
    if normalized == "interaction_done":
        normalized = "not_interaction_ready"

    if normalized not in {
        "interaction_ready",
        "not_interaction_ready",
        "interaction_ongoing",
    }:
        raise ValueError(f"Unsupported interaction-readiness label: {label!r}")

    return normalized  # type: ignore[return-value]


def resolve_image_path(path_value: str) -> Path:
    path = Path(str(path_value).strip())
    if not path.is_absolute() and IMAGE_BASE_DIRECTORY is not None:
        path = IMAGE_BASE_DIRECTORY / path
    return path


def load_frame_image(frame_path_value: Any) -> Image.Image:
    """Load a frame directly from the CSV's frame_path column."""
    resolved = resolve_image_path(str(frame_path_value))
    if not resolved.exists():
        raise FileNotFoundError(f"Frame image does not exist: {resolved}")
    return Image.open(resolved).convert("RGB")


def build_frame_groups(
    dataset: pd.DataFrame,
) -> list[tuple[tuple[Any, Any, Any], pd.DataFrame]]:
    """
    Group person-level rows into unique frames.

    The final CSV still has one row per person, so a frame containing multiple
    people appears multiple times. Qwen should run once per unique frame.
    """
    group_columns = [
        SOURCE_DATASET_COLUMN,
        VIDEO_NAME_COLUMN,
        FRAME_NUMBER_COLUMN,
    ]

    groups = list(dataset.groupby(group_columns, sort=False, dropna=False))
    if MAX_FRAMES is not None:
        groups = groups[:MAX_FRAMES]
    return groups


def ground_truth_people(frame_rows: pd.DataFrame) -> list[dict[str, Any]]:
    people: list[dict[str, Any]] = []

    for _, row in frame_rows.iterrows():
        person_id = normalize_scalar(row[PERSON_ID_COLUMN])
        original_label = str(row[PERSON_LABEL_COLUMN]).strip().lower()

        people.append(
            {
                "person_id": str(person_id),
                "bbox_xyxy": normalize_bbox(row[BBOX_COLUMN]),
                "label_original": original_label,
                "label_3class": normalize_label(original_label),
            }
        )

    return people


def ground_truth_frame_label(frame_rows: pd.DataFrame) -> InteractionLabel:
    """Return the single frame_label shared by all person rows in this frame."""
    label = frame_rows.iloc[0][FRAME_LABEL_COLUMN]
    return normalize_label(str(label))


# =============================================================================
# Model loading and inference
# =============================================================================

def load_model_and_processor(
    model_name: str,
    load_in_4bit: bool,
) -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit quantization requires a CUDA GPU.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            base_model,
            LORA_CHECKPOINT,
        )

        model.eval()
        
    elif torch.cuda.is_available():
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            base_model,
            LORA_CHECKPOINT,
        )

        model.eval()
    else:
        print(
            "Warning: CUDA was not detected. Qwen3-VL inference on CPU will "
            "be extremely slow and may require substantial RAM."
        )
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            base_model,
            LORA_CHECKPOINT,
        )

        model.eval()

    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def infer_processed_image_size(
    processor_inputs: dict[str, Any],
) -> tuple[int, int] | None:
    image_grid_thw = processor_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return None

    if isinstance(image_grid_thw, torch.Tensor):
        grid = image_grid_thw[0].detach().cpu().tolist()
    else:
        grid = image_grid_thw[0]

    if len(grid) != 3:
        return None

    _, grid_height, grid_width = grid
    
    # coordinates debugging: correction 1
    spatial_factor = 32
    return int(grid_width * spatial_factor), int(grid_height * spatial_factor)


def run_inference(
    image: Image.Image,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
) -> tuple[str, tuple[int, int] | None]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    processed_image_size = infer_processed_image_size(inputs)
    model_device = next(model.parameters()).device

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(model_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, input_length:]
    print(f"Generated tokens: {generated_only.shape[1]} / {MAX_NEW_TOKENS}")
    raw_response = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return raw_response, processed_image_size


# =============================================================================
# Model-output parsing and coordinate conversion
# =============================================================================

def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object was found in the model response.")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            print("\n--- Last 1000 characters of malformed response ---")
            print(cleaned[-1000:])
            print("-----------------------------------------------")

            raise ValueError(
                f"The model response contained malformed JSON: {exc}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("The parsed model response is not a JSON object.")
    return parsed


def validate_predictions(response: dict[str, Any]) -> list[PersonPrediction]:
    people = response.get("people")
    if people is None:
        raise ValueError('The response has no "people" field.')
    if not isinstance(people, list):
        raise ValueError('"people" must be a list.')

    validated: list[PersonPrediction] = []
    for index, person in enumerate(people, start=1):
        if not isinstance(person, dict):
            print(f"Skipping prediction {index}: not a JSON object.")
            continue

        bbox = person.get("bbox")
        label = person.get("label")
        person_id = person.get("person_id", index)

        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"Skipping prediction {index}: invalid box {bbox!r}.")
            continue

        try:
            normalized_bbox = [int(round(float(value))) for value in bbox]
            normalized_label = normalize_label(str(label))
            normalized_person_id = int(person_id)
        except (TypeError, ValueError) as exc:
            print(f"Skipping prediction {index}: {exc}")
            continue

        validated.append(
            {
                "person_id": normalized_person_id,
                "bbox": normalized_bbox,
                "label": normalized_label,
            }
        )
        
    # enforce returning only 5 people with biggest bounding boxes if there are more than 5 people detected
    if len(validated) > 5:
        validated.sort(
            key=lambda p: (
                (p["bbox"][2] - p["bbox"][0])
                * (p["bbox"][3] - p["bbox"][1])
            ),
            reverse=True,
        )

        return validated[:5]
        
    return validated


def rescale_predictions(
    predictions: list[PersonPrediction],
    original_size: tuple[int, int],
    processed_size: tuple[int, int] | None,
) -> list[PersonPrediction]:
    original_width, original_height = original_size

    if COORDINATE_MODE == "original":
        scale_x = scale_y = 1.0
    elif COORDINATE_MODE == "normalized_1000":
        scale_x = original_width / 1000.0
        scale_y = original_height / 1000.0
    elif COORDINATE_MODE == "processed":
        if processed_size is None:
            raise ValueError(
                "Qwen's processed dimensions could not be determined. Set "
                "COORDINATE_MODE to 'original' or 'normalized_1000'."
            )
        processed_width, processed_height = processed_size
        scale_x = original_width / processed_width
        scale_y = original_height / processed_height
    else:
        raise ValueError(f"Unknown COORDINATE_MODE: {COORDINATE_MODE}")

    rescaled: list[PersonPrediction] = []
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["bbox"]
        left, right = sorted((round(x1 * scale_x), round(x2 * scale_x)))
        top, bottom = sorted((round(y1 * scale_y), round(y2 * scale_y)))

        left = max(0, min(left, original_width - 1))
        right = max(0, min(right, original_width - 1))
        top = max(0, min(top, original_height - 1))
        bottom = max(0, min(bottom, original_height - 1))

        if right <= left or bottom <= top:
            print(
                f"Skipping person {prediction['person_id']}: box has zero "
                "area after clipping."
            )
            continue

        rescaled.append(
            {
                "person_id": prediction["person_id"],
                "bbox": [left, top, right, bottom],
                "label": prediction["label"],
            }
        )

    return rescaled


# =============================================================================
# Saving predictions and visualizations
# =============================================================================

def safe_filename_part(value: Any) -> str:
    text = str(normalize_scalar(value))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def annotate_image(
    image: Image.Image,
    predictions: list[PersonPrediction],
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except OSError:
        font = ImageFont.load_default()

    line_width = max(2, round(min(image.size) / 300))
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["bbox"]
        text = f"person_{prediction['person_id']}: {prediction['label']}"

        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=line_width)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            [(x1, text_y), (x1 + text_width + 6, text_y + text_height + 6)],
            fill="red",
        )
        draw.text((x1 + 3, text_y + 3), text, fill="white", font=font)

    return annotated


def load_completed_keys(predictions_path: Path) -> set[tuple[str, str, str]]:
    completed: set[tuple[str, str, str]] = set()
    if not predictions_path.exists() or OVERWRITE:
        return completed

    with predictions_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if (
                    record.get("image_error") is None
                    and record.get("parse_error") is None
                ):
                completed.add(
                    (
                        str(record["source_dataset"]),
                        str(record["video_name"]),
                        str(record["frame_number"]),
                    )
                )
    return completed


def save_record(predictions_path: Path, record: dict[str, Any]) -> None:
    with predictions_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def preview_first_frames(frame_groups: list[tuple[tuple[Any, Any, Any], pd.DataFrame]]) -> None:
    """Display the first few unique frames to verify frame_path loading."""
    if PREVIEW_FIRST_N <= 0:
        return

    print(f"\nPreviewing first {min(PREVIEW_FIRST_N, len(frame_groups))} unique frames...")
    for index, (group_key, frame_rows) in enumerate(
        frame_groups[:PREVIEW_FIRST_N], start=1
    ):
        source_dataset, video_name, frame_number = group_key
        image = load_frame_image(frame_rows.iloc[0][FRAME_PATH_COLUMN])

        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(
            f"Preview {index}\n"
            f"{source_dataset} / {video_name} / frame {frame_number}\n"
            f"frame_label={frame_rows.iloc[0][FRAME_LABEL_COLUMN]}, "
            f"people={len(frame_rows)}"
        )
        plt.axis("off")
        plt.show()


# =============================================================================
# Main loop
# =============================================================================

def main() -> None:


    if COORDINATE_MODE not in VALID_COORDINATE_MODES:
        raise ValueError(
            f"COORDINATE_MODE must be one of {sorted(VALID_COORDINATE_MODES)}."
        )

    dataset_all = load_csv_dataset(INPUT_CSV)

    total_rows = len(dataset_all)
    total_frames = dataset_all[
        [SOURCE_DATASET_COLUMN, VIDEO_NAME_COLUMN, FRAME_NUMBER_COLUMN]
    ].drop_duplicates().shape[0]
    total_videos = dataset_all[
        [SOURCE_DATASET_COLUMN, VIDEO_NAME_COLUMN]
    ].drop_duplicates().shape[0]

    dataset = filter_dataset_to_split(
        dataset=dataset_all,
        requested_split=INFERENCE_SPLIT,
    )

    split_frames = dataset[
        [SOURCE_DATASET_COLUMN, VIDEO_NAME_COLUMN, FRAME_NUMBER_COLUMN]
    ].drop_duplicates().shape[0]
    split_videos = dataset[
        [SOURCE_DATASET_COLUMN, VIDEO_NAME_COLUMN]
    ].drop_duplicates().shape[0]

    print(
        f"Loaded {total_rows:,} person rows, {total_frames:,} unique frames, "
        f"and {total_videos:,} videos."
    )
    print(
        f"Using {INFERENCE_SPLIT!r} split: {len(dataset):,} person rows, "
        f"{split_frames:,} unique frames, {split_videos:,} videos."
    )

    frame_groups = build_frame_groups(dataset)
    preview_first_frames(frame_groups)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUTPUT_DIR / "raw_responses"
    annotated_dir = OUTPUT_DIR / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_ANNOTATED:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "predictions.jsonl"
    if OVERWRITE and predictions_path.exists():
        predictions_path.unlink()

    completed_keys = load_completed_keys(predictions_path)

    print(f"\nPerson-level rows in selected split: {len(dataset):,}")
    print(f"Unique frames to process: {len(frame_groups):,}")
    print(f"Loading model: {MODEL_NAME}")

    model, processor = load_model_and_processor(MODEL_NAME, LOAD_IN_4BIT)

    for frame_index, (group_key, frame_rows) in enumerate(frame_groups, start=1):
        source_dataset, video_name, frame_number = [
            normalize_scalar(value) for value in group_key
        ]
        record_key = (str(source_dataset), str(video_name), str(frame_number))

        if record_key in completed_keys:
            print(
                f"[{frame_index}/{len(frame_groups)}] Skipping completed "
                f"{source_dataset}/{video_name}/{frame_number}"
            )
            continue

        print(
            f"[{frame_index}/{len(frame_groups)}] Processing "
            f"{source_dataset}/{video_name}/{frame_number}"
        )

        frame_path = str(frame_rows.iloc[0][FRAME_PATH_COLUMN])
        ground_truth = ground_truth_people(frame_rows)
        gt_frame_label = ground_truth_frame_label(frame_rows)

        image_error: str | None = None
        parse_error: str | None = None
        predictions: list[PersonPrediction] = []
        processed_size: tuple[int, int] | None = None
        raw_response = ""
        image: Image.Image | None = None
        image_width: int | None = None
        image_height: int | None = None

        try:
            image = load_frame_image(frame_path)
            image_width, image_height = image.size
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            image_error = str(exc)
            print(f"Image loading failed: {exc}")

        if image is not None:
            try:
                raw_response, processed_size = run_inference(
                    image=image,
                    model=model,
                    processor=processor,
                )

                response_json = extract_json_object(raw_response)
                predictions = validate_predictions(response_json)
                predictions = rescale_predictions(
                    predictions=predictions,
                    original_size=image.size,
                    processed_size=processed_size,
                )
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                parse_error = str(exc)
                print(f"Inference/parsing failed: {exc}")

        safe_stem = "__".join(
            safe_filename_part(value)
            for value in (source_dataset, video_name, frame_number)
        )

        raw_path: Path | None = None
        if raw_response:
            raw_path = raw_dir / f"{safe_stem}.txt"
            raw_path.write_text(raw_response, encoding="utf-8")

        record: dict[str, Any] = {
            "source_dataset": source_dataset,
            "video_name": video_name,
            "frame_number": frame_number,
            "frame_path": frame_path,
            "image_width": image_width,
            "image_height": image_height,
            "model": MODEL_NAME,
            "dataset_split": INFERENCE_SPLIT,
            "coordinate_mode": COORDINATE_MODE,
            "processed_image_size": (
                {"width": processed_size[0], "height": processed_size[1]}
                if processed_size is not None
                else None
            ),
            "ground_truth_frame_label": gt_frame_label,
            "ground_truth_people": ground_truth,
            "predicted_people": predictions,
            "image_error": image_error,
            "parse_error": parse_error,
            "raw_response_path": str(raw_path) if raw_path else None,
        }
        save_record(predictions_path, record)

        if SAVE_ANNOTATED and image is not None:
            annotated = annotate_image(image, predictions)
            annotated.save(annotated_dir / f"{safe_stem}.jpg", quality=95)

        if image is not None:
            image.close()

    print(f"\nPredictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
