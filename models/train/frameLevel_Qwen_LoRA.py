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

import ast
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

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
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/"
    "balanced_frame_based_dataset.csv"
)

OUTPUT_DIR = Path(
    "/home/marilyn/Downloads/Interaction-Readiness-Detection-Model-main/"
    "outputs/Qwen3-VL-8B-LoRA-frameLevel"
)

# Change this if your column is called dataset_split, set, etc.
SPLIT_COLUMN = "split"

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LOGGING_STEPS = 10
SAVE_STEPS = 250

SEED = 42

# Image resolution.
#
# Qwen documentation stresses that training image resolution
# matters substantially. These values limit excessive VRAM use.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


# ============================================================
# PROMPT
# ============================================================

# IMPORTANT:
# Ideally replace this with EXACTLY the prompt used in your
# frameLevel_ZeroShot_predict.py script.
#
# Fine-tuning and zero-shot evaluation should otherwise have the
# same task definition.

PROMPT = """
Analyze the image and identify the people visible in the scene.

For each person, return:
1. Their bounding box as [x1, y1, x2, y2].
2. Their interaction-readiness label.

The possible labels are:
- interaction_ready
- interaction_ongoing
- not_interaction_ready

Definitions:
- interaction_ready: the person appears ready or intending to interact with the robot.
- interaction_ongoing: the person is currently interacting with the robot.
- not_interaction_ready: the person is not currently ready to interact with the robot.

Return ONLY valid JSON in the following format:

[
    {
        "bbox": [x1, y1, x2, y2],
        "label": "interaction_ready"
    }
]

Do not include explanations or text outside the JSON.
"""


# ============================================================
# UTILITIES
# ============================================================

def parse_bbox(value):
    """
    Convert a bbox stored in the CSV to:
        [x1, y1, x2, y2]

    Handles:
        "[10, 20, 100, 200]"
        [10, 20, 100, 200]
    """

    if isinstance(value, str):
        value = ast.literal_eval(value)

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Invalid bbox: {value}")

    if len(value) != 4:
        raise ValueError(
            f"Expected bbox with 4 coordinates, received: {value}"
        )

    return [round(float(x), 2) for x in value]


def normalize_label(label):
    """
    Normalize labels for the benchmark.

    interaction_done is treated as not_interaction_ready,
    consistent with the rest of the evaluation pipeline.
    """

    label = str(label).strip()

    if label == "interaction_done":
        return "not_interaction_ready"

    valid_labels = {
        "interaction_ready",
        "interaction_ongoing",
        "not_interaction_ready",
    }

    if label not in valid_labels:
        raise ValueError(f"Unknown label: {label}")

    return label


def build_target(frame_df):
    """
    Construct the assistant target for one entire frame.
    """

    people = []

    for _, row in frame_df.iterrows():

        person = {
            "bbox": parse_bbox(row["bbox"]),
            "label": normalize_label(row["label"]),
        }

        people.append(person)

    # Compact JSON saves tokens and makes output formatting easier
    return json.dumps(
        people,
        separators=(",", ":"),
        ensure_ascii=False,
    )


# ============================================================
# BUILD FRAME-LEVEL DATASET
# ============================================================

def build_frame_dataframe(df):
    """
    Convert person-level rows into one row per frame.

    Original:
        frame 10 / person 1
        frame 10 / person 2
        frame 10 / person 3

    Becomes:
        frame 10 -> image + JSON containing persons 1,2,3
    """

    required_columns = {
        "source_dataset",
        "video_id",
        "frame_id",
        "image",
        "bbox",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    group_columns = [
        "source_dataset",
        "video_id",
        "frame_id",
    ]

    frames = []

    grouped = df.groupby(
        group_columns,
        sort=False,
        dropna=False,
    )

    for frame_key, frame_df in grouped:

        # Every person belonging to this frame should point
        # to the same image.
        image_paths = frame_df["image"].dropna().unique()

        if len(image_paths) != 1:
            raise ValueError(
                f"Frame {frame_key} contains "
                f"{len(image_paths)} different image paths."
            )

        image_path = str(image_paths[0])

        frames.append(
            {
                "source_dataset": frame_key[0],
                "video_id": frame_key[1],
                "frame_id": frame_key[2],
                "image": image_path,
                "target": build_target(frame_df),
            }
        )

    return pd.DataFrame(frames)


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

        return {
            "image": row["image"],
            "target": row["target"],
        }


# ============================================================
# DATA COLLATOR
# ============================================================

class Qwen3VLCollator:
    """
    Creates multimodal Qwen training inputs.

    Only tokens belonging to the assistant's answer contribute
    to the language-model loss.

    The user prompt and image tokens are masked with -100.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):

        # We deliberately train with per_device_batch_size=1.
        #
        # Multimodal batching becomes more complicated because
        # images can contain different numbers of vision tokens.
        if len(examples) != 1:
            raise ValueError(
                "This collator currently expects batch_size=1. "
                "Use gradient accumulation for a larger effective batch."
            )

        example = examples[0]

        image_path = example["image"]
        target = example["target"]

        # ----------------------------------------------------
        # USER PROMPT
        # ----------------------------------------------------

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

        # Tokenize prompt only.
        #
        # add_generation_prompt=True adds the beginning of the
        # assistant turn.
        prompt_inputs = self.processor.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        prompt_length = prompt_inputs["input_ids"].shape[1]

        # ----------------------------------------------------
        # COMPLETE TRAINING CONVERSATION
        # ----------------------------------------------------

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

        # Some processor versions produce token_type_ids even
        # though the Qwen model does not need the standard field.
        inputs.pop("token_type_ids", None)

        # ----------------------------------------------------
        # LABEL MASK
        # ----------------------------------------------------

        labels = inputs["input_ids"].clone()

        # Do not calculate loss on:
        #   image tokens
        #   system/user prompt
        #   beginning of assistant turn
        labels[:, :prompt_length] = -100

        # Ignore padding if present
        if self.processor.tokenizer.pad_token_id is not None:
            labels[
                inputs["input_ids"]
                == self.processor.tokenizer.pad_token_id
            ] = -100

        inputs["labels"] = labels

        return inputs


# ============================================================
# MODEL
# ============================================================

def load_model():

    print(f"\nLoading model: {MODEL_NAME}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Important for gradient checkpointing
    model.config.use_cache = False

    # --------------------------------------------------------
    # Freeze base model
    # --------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # LoRA
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

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD CSV
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(CSV_PATH)

    print(f"Person-level rows: {len(df):,}")

    print("\nColumns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    if SPLIT_COLUMN not in df.columns:
        raise ValueError(
            f"\nCould not find split column '{SPLIT_COLUMN}'.\n"
            f"Available columns:\n{df.columns.tolist()}\n\n"
            "Do NOT randomly split frames here if your benchmark "
            "already has predefined train/val/test video splits."
        )

    train_person_df = df[
        df[SPLIT_COLUMN] == TRAIN_SPLIT
    ].copy()

    val_person_df = df[
        df[SPLIT_COLUMN] == VAL_SPLIT
    ].copy()

    print(
        f"\nTrain person annotations: "
        f"{len(train_person_df):,}"
    )

    print(
        f"Validation person annotations: "
        f"{len(val_person_df):,}"
    )

    # --------------------------------------------------------
    # GROUP PEOPLE BY FRAME
    # --------------------------------------------------------

    print("\nGrouping annotations by frame...")

    train_frames = build_frame_dataframe(
        train_person_df
    )

    val_frames = build_frame_dataframe(
        val_person_df
    )

    print(
        f"Training frames: "
        f"{len(train_frames):,}"
    )

    print(
        f"Validation frames: "
        f"{len(val_frames):,}"
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print("\nExample training sample:")
    print(
        json.dumps(
            train_frames.iloc[0].to_dict(),
            indent=2,
        )
    )

    # Save grouped data for inspection
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

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = InteractionReadinessDataset(
        train_frames
    )

    eval_dataset = InteractionReadinessDataset(
        val_frames
    )

    collator = Qwen3VLCollator(
        processor
    )

    # --------------------------------------------------------
    # MODEL + LoRA
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=str(OUTPUT_DIR),

        # --------------------
        # epochs
        # --------------------

        num_train_epochs=NUM_EPOCHS,

        # --------------------
        # batches
        # --------------------

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # --------------------
        # optimizer
        # --------------------

        learning_rate=LEARNING_RATE,

        weight_decay=0.01,

        warmup_ratio=0.03,

        lr_scheduler_type="cosine",

        max_grad_norm=1.0,

        # --------------------
        # precision
        # --------------------

        bf16=True,
        fp16=False,

        # --------------------
        # memory
        # --------------------

        gradient_checkpointing=True,

        # --------------------
        # evaluation
        # --------------------

        eval_strategy="steps",
        eval_steps=SAVE_STEPS,

        # --------------------
        # logging
        # --------------------

        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,

        # --------------------
        # checkpoints
        # --------------------

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # --------------------
        # misc
        # --------------------

        remove_unused_columns=False,

        dataloader_num_workers=0,

        report_to="none",

        seed=SEED,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nStarting LoRA fine-tuning...\n")

    trainer.train()

    # --------------------------------------------------------
    # SAVE LoRA ADAPTER
    # --------------------------------------------------------

    print("\nSaving LoRA adapter...")

    trainer.save_model(
        str(OUTPUT_DIR / "final_adapter")
    )

    processor.save_pretrained(
        OUTPUT_DIR / "final_adapter"
    )

    print(
        "\nTraining finished."
        f"\nAdapter saved to:"
        f"\n{OUTPUT_DIR / 'final_adapter'}"
    )


if __name__ == "__main__":
    main()