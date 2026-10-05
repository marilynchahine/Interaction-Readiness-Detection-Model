"""
Frame-level LoRA fine-tuning for Qwen3-VL
Interaction Readiness Detection benchmark

Input CSV structure expected:
    source_dataset
    video_id
    person_id
    frame_id
    image
    bbox
    label
    split

Each CSV row corresponds to one person.

IMPORTANT:
Rows are GROUPED BY FRAME before training, because Qwen must learn:
    image -> all people + bbox + interaction-readiness label

Example target:
[
    {
        "bbox": [120, 84, 310, 470],
        "label": "interaction_ready"
    },
    {
        "bbox": [420, 102, 590, 465],
        "label": "not_interaction_ready"
    }
]
"""

import random
import ast
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

CSV_PATH = Path(
    "/home/marilyn/Downloads/data/balanced_frame_based_dataset_linux.csv"
)

OUTPUT_DIR = Path(
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/outputs/resized/Qwen3VL_8B_Instruct_LoRA"
)


# ------------------------------------------------------------
# Dataset splits
# ------------------------------------------------------------

TRAIN_SPLIT = "Train"
VAL_SPLIT = "Val"


# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

NUM_EPOCHS = 2

LEARNING_RATE = 1e-5

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8


# ------------------------------------------------------------
# Logging / validation / checkpointing
# ------------------------------------------------------------

LOGGING_STEPS = 10

EVAL_STEPS = 250
SAVE_STEPS = 250

SAVE_TOTAL_LIMIT = 3


# ------------------------------------------------------------
# Image resolution
# ------------------------------------------------------------

# Qwen3-VL uses dynamic image resolution.
#
# 256 * 28 * 28 = 200,704 pixels
# 1024 * 28 * 28 = 802,816 pixels
#
# Aspect ratio is preserved.

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42


# ============================================================
# PROMPT
# ============================================================

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


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# ============================================================
# CUDA / GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This script is configured to run on GPU only."
    )

CUDA_DEVICE = 0
torch.cuda.set_device(CUDA_DEVICE)
device = torch.device(f"cuda:{CUDA_DEVICE}")

print("\n========================================")
print("CUDA configuration")
print("========================================")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device:    {device}")
print(f"GPU:            {torch.cuda.get_device_name(CUDA_DEVICE)}")


# ============================================================
# BOUNDING BOX PARSING
# ============================================================

def parse_bbox(value):
    """
    Parse the bounding box stored in the CSV.

    Dataset format:

        [x, y, width, height]

    Example:

        [664.04, 1.55, 55.62, 105.71]
    """

    if isinstance(value, str):
        value = ast.literal_eval(value)

    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"Bounding box must be list/tuple. Received: {value}"
        )

    if len(value) != 4:
        raise ValueError(
            f"Bounding box must contain exactly 4 values. "
            f"Received: {value}"
        )

    bbox = [round(float(v), 2) for v in value]

    return bbox

def bbox_xywh_to_xyxy(value):
    """Convert [x, y, width, height] -> [x1, y1, x2, y2]."""

    x, y, width, height = parse_bbox(value)

    return [
        round(x, 2),
        round(y, 2),
        round(x + width, 2),
        round(y + height, 2),
    ]

# ============================================================
# LABEL CHECK
# ============================================================

def validate_label(label):
    """
    Validate the person-level class.
    """

    label = str(label).strip()

    valid_labels = {
        "not_interaction_ready",
        "interaction_ready",
        "interaction_ongoing",
    }

    if label not in valid_labels:
        raise ValueError(
            f"Unknown person label: {label}"
        )

    return label


# ============================================================
# TARGET CREATION
# ============================================================

def build_target(frame_df):
    """
    Create the assistant's expected JSON output for one frame.
    Bounding boxes are converted from CSV xywh to xyxy.
    """

    people = []

    for person_idx, (_, row) in enumerate(frame_df.iterrows(), start=1):

        bbox = bbox_xywh_to_xyxy(row["bbox"])
        label = validate_label(row["person_label"])

        people.append(
            {
                "person_id": person_idx,
                "bbox": bbox,
                "label": label,
            }
        )

    target = {
        "people": people
    }

    return json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ============================================================
# GROUP PERSON ROWS INTO FRAMES
# ============================================================

def build_frame_dataframe(df):
    """
    Convert the person-level CSV into a frame-level dataset.

    Original CSV:

        frame 10 | person 0 | bbox A | label A
        frame 10 | person 1 | bbox B | label B
        frame 11 | person 0 | bbox C | label C

    Becomes:

        frame 10 | image | [person A, person B]
        frame 11 | image | [person C]
    """

    required_columns = {
        "source_dataset",
        "video_name",
        "frame_number",
        "frame_label",
        "person_id",
        "person_label",
        "bbox",
        "frame_path",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    frame_rows = []

    group_columns = [
        "source_dataset",
        "video_name",
        "frame_number",
    ]

    grouped = df.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )

    for frame_key, frame_df in grouped:

        source_dataset = frame_key[0]
        video_name = frame_key[1]
        frame_number = frame_key[2]

        # ----------------------------------------------------
        # Verify image path
        # ----------------------------------------------------

        frame_paths = (
            frame_df["frame_path"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(frame_paths) != 1:
            raise ValueError(
                f"{frame_key}: expected exactly one frame_path, "
                f"found {len(frame_paths)}."
            )

        frame_path = frame_paths[0]

        # ----------------------------------------------------
        # Verify frame-level label consistency
        # ----------------------------------------------------

        frame_labels = (
            frame_df["frame_label"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(frame_labels) != 1:
            raise ValueError(
                f"{frame_key}: inconsistent frame_label values: "
                f"{frame_labels}"
            )

        frame_label = frame_labels[0]

        # ----------------------------------------------------
        # Build complete person target
        # ----------------------------------------------------

        target = build_target(frame_df)

        frame_rows.append(
            {
                "source_dataset": source_dataset,
                "video_name": video_name,
                "frame_number": int(frame_number),
                "frame_label": frame_label,
                "frame_path": frame_path,
                "target": target,
            }
        )

    frame_dataframe = pd.DataFrame(frame_rows)

    return frame_dataframe


# ============================================================
# PYTORCH DATASET
# ============================================================

class InteractionReadinessDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = str(row["frame_path"])

        if not Path(image_path).exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        return {
            "image": image_path,
            "target": row["target"],
        }


# ============================================================
# MULTIMODAL COLLATOR
# ============================================================

class Qwen3VLCollator:
    """
    Converts one image + expected JSON response into Qwen3-VL
    multimodal training tensors.

    Loss is computed ONLY on the assistant response.

    The following are masked:
        - image tokens
        - user prompt
        - assistant-turn prefix

    Masked tokens receive label = -100.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):

        # We intentionally use batch size 1 because multimodal
        # training is memory-intensive.
        #
        # Effective batch size is increased through gradient
        # accumulation.

        if len(examples) != 1:
            raise ValueError(
                "This collator expects per_device_batch_size=1."
            )

        example = examples[0]

        image_path = example["image"]
        target = example["target"]

        # ====================================================
        # PROMPT-ONLY CONVERSATION
        # ====================================================

        user_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            }
        ]

        # Tokenize prompt + image + beginning of assistant turn.
        #
        # We need its length so that all of these tokens can
        # later be masked from the loss.

        prompt_inputs = self.processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        prompt_length = prompt_inputs["input_ids"].shape[1]

        # ====================================================
        # COMPLETE TRAINING CONVERSATION
        # ====================================================

        full_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": MIN_PIXELS,
                        "max_pixels": MAX_PIXELS,
                    },
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": target,
                    }
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        # Qwen processor may produce this field, but it is not
        # needed by the model here.
        inputs.pop("token_type_ids", None)

        # ====================================================
        # LABEL MASKING
        # ====================================================

        labels = inputs["input_ids"].clone()

        # Ignore:
        # image tokens
        # prompt tokens
        # beginning of assistant message

        labels[:, :prompt_length] = -100

        # Ignore padding
        pad_token_id = self.processor.tokenizer.pad_token_id

        if pad_token_id is not None:
            labels[
                inputs["input_ids"] == pad_token_id
            ] = -100

        inputs["labels"] = labels

        return inputs


# ============================================================
# LOAD QWEN + APPLY LoRA
# ============================================================

def load_model():

    print("\n========================================")
    print("Loading base model")
    print("========================================")

    print(MODEL_NAME)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Explicitly place the complete base model on the selected GPU.
    model = model.to(device)

    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError(
            f"Model was not loaded on CUDA. Current device: "
            f"{next(model.parameters()).device}"
        )

    print(
        f"Model loaded on: {next(model.parameters()).device}"
    )

    # Cache is useful for inference but should be disabled while
    # using gradient checkpointing during training.
    model.config.use_cache = False

    # --------------------------------------------------------
    # Freeze ALL pretrained parameters
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------------

    lora_config = LoraConfig(

        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        bias="none",

        task_type=TaskType.CAUSAL_LM,
    )

    # --------------------------------------------------------
    # Insert adapters
    # --------------------------------------------------------

    model = get_peft_model(
        model,
        lora_config,
    )

    print("\n========================================")
    print("LoRA trainable parameters")
    print("========================================")

    model.print_trainable_parameters()

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD CSV
    # ========================================================

    print("\n========================================")
    print("Loading CSV")
    print("========================================")

    print(f"CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"\nTotal person-level rows: {len(df):,}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSplit distribution:")
    print(
        df["split"]
        .value_counts(dropna=False)
    )

    # ========================================================
    # REMOVE ROWS WITHOUT SPLIT
    # ========================================================

    missing_split_count = df["split"].isna().sum()

    if missing_split_count > 0:

        print(
            f"\nIgnoring {missing_split_count} rows "
            "with no split assigned."
        )

        df = df[
            df["split"].notna()
        ].copy()

    # ========================================================
    # CREATE TRAIN / VALIDATION PERSON DATA
    # ========================================================

    train_person_df = df[
        df["split"] == TRAIN_SPLIT
    ].copy()

    val_person_df = df[
        df["split"] == VAL_SPLIT
    ].copy()

    if len(train_person_df) == 0:
        raise ValueError(
            f"No rows found for split '{TRAIN_SPLIT}'."

        )

    if len(val_person_df) == 0:
        raise ValueError(
            f"No rows found for split '{VAL_SPLIT}'."
        )

    print(
        f"\nTrain person annotations: "
        f"{len(train_person_df):,}"
    )

    print(
        f"Validation person annotations: "
        f"{len(val_person_df):,}"
    )

    # ========================================================
    # GROUP PEOPLE BY FRAME
    # ========================================================

    print("\n========================================")
    print("Grouping annotations by frame")
    print("========================================")

    train_frames = build_frame_dataframe(
        train_person_df
    )

    val_frames = build_frame_dataframe(
        val_person_df
    )

    print(
        f"\nTraining frames: "
        f"{len(train_frames):,}"
    )

    print(
        f"Validation frames: "
        f"{len(val_frames):,}"
    )

    # ========================================================
    # SANITY CHECK LABELS
    # ========================================================

    print("\nTraining person-label distribution:")

    print(
        train_person_df["person_label"]
        .value_counts()
    )

    print("\nTraining frame-label distribution:")

    print(
        train_frames["frame_label"]
        .value_counts()
    )

    # ========================================================
    # SHOW ONE TRAINING EXAMPLE
    # ========================================================

    print("\n========================================")
    print("Example training item")
    print("========================================")

    example = train_frames.iloc[0]

    print(f"Dataset: {example['source_dataset']}")
    print(f"Video:   {example['video_name']}")
    print(f"Frame:   {example['frame_number']}")
    print(f"Image:   {example['frame_path']}")

    print("\nTarget:")

    print(
        json.dumps(
            json.loads(example["target"]),
            indent=2,
        )
    )

    # ========================================================
    # VERIFY IMAGE FILES
    # ========================================================

    print("\n========================================")
    print("Checking image paths")
    print("========================================")

    # Check a subset here rather than opening every image.
    sample_paths = train_frames[
        "frame_path"
    ].head(100)

    missing_paths = [
        path
        for path in sample_paths
        if not Path(path).exists()
    ]

    if missing_paths:

        print("\nExample missing paths:")

        for path in missing_paths[:10]:
            print(path)

        raise FileNotFoundError(
            "\nSome frame paths do not exist. "
            "Check the frame_path column before training."
        )

    print(
        "First 100 training image paths are valid."
    )

    # ========================================================
    # SAVE GROUPED DATASET FOR INSPECTION
    # ========================================================

    train_frames.to_json(
        OUTPUT_DIR / "train_frames.json",
        orient="records",
        indent=2,
    )

    val_frames.to_json(
        OUTPUT_DIR / "val_frames.json",
        orient="records",
        indent=2,
    )

    # ========================================================
    # PROCESSOR
    # ========================================================

    print("\n========================================")
    print("Loading processor")
    print("========================================")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
    )

    # ========================================================
    # PYTORCH DATASETS
    # ========================================================

    train_dataset = InteractionReadinessDataset(
        train_frames
    )

    eval_dataset = InteractionReadinessDataset(
        val_frames
    )

    # ========================================================
    # COLLATOR
    # ========================================================

    collator = Qwen3VLCollator(
        processor
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = TrainingArguments(

        output_dir=str(OUTPUT_DIR),

        # ----------------------------------------------------
        # Training duration
        # ----------------------------------------------------

        num_train_epochs=NUM_EPOCHS,

        # ----------------------------------------------------
        # Batch sizes
        # ----------------------------------------------------

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        learning_rate=LEARNING_RATE,

        weight_decay=0.01,

        warmup_steps=450,

        lr_scheduler_type="cosine",

        max_grad_norm=1.0,

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        bf16=True,
        fp16=False,

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        gradient_checkpointing=True,

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        eval_strategy="steps",

        eval_steps=EVAL_STEPS,

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logging_strategy="steps",

        logging_steps=LOGGING_STEPS,

        # ----------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------

        save_strategy="steps",

        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        # ----------------------------------------------------
        # Dataset handling
        # ----------------------------------------------------

        remove_unused_columns=False,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        # ----------------------------------------------------
        # External logging
        # ----------------------------------------------------

        report_to="none",

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        seed=SEED,
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n========================================")
    print("Starting LoRA fine-tuning")
    print("========================================\n")

    # ========================================================
    # RESUME FROM LATEST CHECKPOINT IF ONE EXISTS
    # ========================================================

    last_checkpoint = None

    if OUTPUT_DIR.exists():
        last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR))

    if last_checkpoint is not None:
        print("\n========================================")
        print("Resuming training from checkpoint")
        print("========================================")
        print(last_checkpoint)

        trainer.train(
            resume_from_checkpoint=last_checkpoint
        )
    else:
        print("\nNo existing checkpoint found.")
        print("Starting training from the base model + new LoRA adapter.")

        trainer.train()

    # ========================================================
    # SAVE FINAL / BEST ADAPTER
    # ========================================================

    final_adapter_dir = (
        OUTPUT_DIR / "final_adapter"
    )

    print("\n========================================")
    print("Saving LoRA adapter")
    print("========================================")

    trainer.save_model(
        str(final_adapter_dir)
    )

    processor.save_pretrained(
        str(final_adapter_dir)
    )

    # Enable cache again for later inference
    model.config.use_cache = True

    print("\nTraining finished.")

    print(
        f"\nLoRA adapter saved to:\n"
        f"{final_adapter_dir}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
