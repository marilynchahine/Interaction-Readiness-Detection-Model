from pathlib import Path
import json
import shutil
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image as PILImage

from datasets import (
    Dataset,
    Features,
    Image as HFImage,
    Sequence,
    Value,
)


"""
Convert Parquet annotation files containing local frame paths into
self-contained Hugging Face-compatible Parquet files.

The generator loads each frame as a real PIL image. Hugging Face then
encodes it into the Parquet-compatible representation:

    struct<
        bytes: binary,
        path: string
    >

The output also contains a label_ID column:

    not_interaction_ready -> 0
    interaction_ready     -> 1
    interaction_ongoing   -> 2
    interaction_done      -> 3

Set MAX_ROWS to:
    10   -> process only the first 10 rows of each file
    100  -> process only the first 100 rows of each file
    None -> process every row
"""


# ============================================================
# Configuration
# ============================================================

DATASET_DIRECTORY = Path(
    r"D:\EngagementDetection\Push_to_HF\parquet_final_public"
)

OUTPUT_DIRECTORY = Path(
    r"D:\EngagementDetection\Push_to_HF\parquet_final_gated"
)

CACHE_DIRECTORY = (
    DATASET_DIRECTORY / "_hf_generator_cache"
)

PARQUET_FILES = [
    DATASET_DIRECTORY / "SSUP-HRI.parquet",
]

# Number of rows to process from each Parquet file.
#
# Examples:
# MAX_ROWS = 10
# MAX_ROWS = 100
# MAX_ROWS = None
#
# Use None to process all rows.
MAX_ROWS = None

# Update the live terminal output every N rows.
PROGRESS_INTERVAL = 100

# Reuse any completed Hugging Face generator cache from a previous run.
# Keep this False to recover the six-hour generation step when possible.
FORCE_REGENERATE_CACHE = False

# Keep the generated Arrow cache after a successful run. This uses disk space,
# but allows the final writing/validation stage to be retried without decoding
# every image again.
KEEP_GENERATOR_CACHE = True

# Number of rows written to Parquet at once. Embedded images are large, so a
# small batch greatly reduces the risk of Arrow offset overflow and high RAM use.
PARQUET_WRITE_BATCH_SIZE = 8


# ============================================================
# Label mapping
# ============================================================

LABEL_TO_ID = {
    "not_interaction_ready": 0,
    "interaction_ready": 1,
    "interaction_ongoing": 2,
    "interaction_done": 3,
}


# ============================================================
# Hugging Face feature definition
# ============================================================

FEATURES = Features(
    {
        "source_dataset": Value("string"),
        "video_id": Value("string"),
        "person_id": Value("string"),
        "frame_id": Value("int64"),

        # The generator yields PIL images for this column.
        "image": HFImage(),

        # Bounding-box format:
        # [x, y, width, height]
        "bbox": Sequence(
            feature=Value("float32"),
            length=4,
        ),

        "label": Value("string"),
        "label_ID": Value("int64"),
    }
)


# ============================================================
# General helpers
# ============================================================

def format_duration(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.
    """

    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_output_path(
    input_path: Path,
) -> Path:
    """
    Generate the output path.

    Limited test files receive a suffix so they do not overwrite
    complete output files.
    """

    if MAX_ROWS is None:
        output_name = input_path.name

    else:
        output_name = (
            f"{input_path.stem}"
            f"_first_{MAX_ROWS}_rows.parquet"
        )

    return OUTPUT_DIRECTORY / output_name


def resolve_image_path(
    image_path_value,
) -> Path:
    """
    Convert a frame_path value into a valid absolute path.

    Relative paths are interpreted relative to DATASET_DIRECTORY.
    """

    if image_path_value is None:
        raise ValueError(
            "The frame_path value is None."
        )

    image_path = Path(
        str(image_path_value)
    )

    if not image_path.is_absolute():
        image_path = (
            DATASET_DIRECTORY / image_path
        )

    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Frame does not exist: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Frame path is not a file: {image_path}"
        )

    return image_path


def load_pil_image(
    image_path: Path,
) -> PILImage.Image:
    """
    Load an image and return an independent RGB PIL image.

    copy() ensures the returned image remains usable after the source
    file has been closed.
    """

    try:
        with PILImage.open(image_path) as image:
            image.load()

            pil_image = (
                image
                .convert("RGB")
                .copy()
            )

    except Exception as error:
        raise ValueError(
            f"Could not decode image: {image_path}"
        ) from error

    if pil_image.width <= 0 or pil_image.height <= 0:
        raise ValueError(
            f"Image has invalid dimensions: {image_path}"
        )

    return pil_image


def normalize_bbox(
    bbox,
    row_number: int,
) -> list[float]:
    """
    Convert a bounding box into a list containing four floats.
    """

    if hasattr(bbox, "tolist"):
        bbox = bbox.tolist()

    if not isinstance(
        bbox,
        (list, tuple),
    ):
        raise ValueError(
            f"Invalid bbox type at row {row_number}: "
            f"{type(bbox).__name__}"
        )

    if len(bbox) != 4:
        raise ValueError(
            f"Invalid bbox at row {row_number}: "
            f"{bbox}"
        )

    try:
        normalized_bbox = [
            float(value)
            for value in bbox
        ]

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Non-numeric bbox at row "
            f"{row_number}: {bbox}"
        ) from error

    return normalized_bbox


def get_label_id(
    label,
    row_number: int,
) -> int:
    """
    Convert a string label into its corresponding integer ID.
    """

    normalized_label = str(label).strip()

    if normalized_label not in LABEL_TO_ID:
        raise ValueError(
            f"Unknown label at row {row_number}: "
            f"{normalized_label!r}\n"
            f"Expected one of: "
            f"{sorted(LABEL_TO_ID.keys())}"
        )

    return LABEL_TO_ID[normalized_label]


def validate_input_columns(
    input_path: Path,
) -> tuple[int, int]:
    """
    Validate the input schema.

    Returns:
        complete_row_count
        rows_to_process
    """

    required_columns = {
        "source_dataset",
        "video_id",
        "person_id",
        "frame_id",
        "frame_path",
        "bbox",
        "label",
    }

    parquet_file = pq.ParquetFile(
        str(input_path)
    )

    # schema_arrow.names returns only top-level columns.
    available_columns = set(
        parquet_file.schema_arrow.names
    )

    missing_columns = (
        required_columns - available_columns
    )

    if missing_columns:
        raise ValueError(
            f"{input_path.name} is missing columns: "
            f"{sorted(missing_columns)}\n"
            f"Available columns: "
            f"{sorted(available_columns)}"
        )

    complete_row_count = (
        parquet_file.metadata.num_rows
    )

    if MAX_ROWS is None:
        rows_to_process = complete_row_count

    else:
        if not isinstance(MAX_ROWS, int):
            raise TypeError(
                "MAX_ROWS must be an integer or None.\n"
                f"Current value: {MAX_ROWS!r}"
            )

        if MAX_ROWS <= 0:
            raise ValueError(
                "MAX_ROWS must be greater than zero "
                "or set to None."
            )

        rows_to_process = min(
            MAX_ROWS,
            complete_row_count,
        )

    return (
        complete_row_count,
        rows_to_process,
    )


# ============================================================
# Progress display
# ============================================================

def print_progress(
    position: int,
    total_rows: int,
    image_path: Path,
    video_id: str,
    person_id: str,
    start_time: float,
) -> None:
    """
    Display live progress on one terminal line.
    """

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    rows_per_second = (
        position / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    remaining_rows = (
        total_rows - position
    )

    estimated_remaining_seconds = (
        remaining_rows / rows_per_second
        if rows_per_second > 0
        else 0.0
    )

    percentage = (
        position / total_rows * 100
        if total_rows > 0
        else 100.0
    )

    print(
        f"\r"
        f"Embedding {position:,}/{total_rows:,} "
        f"({percentage:6.2f}%) | "
        f"Video: {video_id} | "
        f"Image: {image_path.name} | "
        f"Person: {person_id} | "
        f"Speed: {rows_per_second:,.2f} rows/s | "
        f"Elapsed: {format_duration(elapsed_seconds)} | "
        f"ETA: {format_duration(estimated_remaining_seconds)}",
        end="",
        flush=True,
    )


# ============================================================
# Row generator
# ============================================================

def generate_rows(
    input_path_string: str,
):
    """
    Yield converted rows.

    Only the required columns are loaded. If MAX_ROWS is not None,
    only that number of rows is loaded from the first Parquet batches.

    Consecutive rows using the same frame reuse the decoded PIL image.
    """

    input_path = Path(
        input_path_string
    )

    parquet_file = pq.ParquetFile(
        str(input_path)
    )

    columns_to_read = [
        "source_dataset",
        "video_id",
        "person_id",
        "frame_id",
        "frame_path",
        "bbox",
        "label",
    ]

    if MAX_ROWS is None:
        rows_to_process = (
            parquet_file.metadata.num_rows
        )
    else:
        rows_to_process = min(
            MAX_ROWS,
            parquet_file.metadata.num_rows,
        )

    start_time = time.perf_counter()

    previous_image_path = None
    previous_pil_image = None

    position = 0

    # iter_batches avoids loading the complete input file into RAM.
    for batch in parquet_file.iter_batches(
        columns=columns_to_read,
        batch_size=1_000,
    ):
        batch_dataframe = batch.to_pandas()

        for row in batch_dataframe.itertuples(
            index=False
        ):
            if (
                MAX_ROWS is not None
                and position >= MAX_ROWS
            ):
                break

            position += 1

            image_path = resolve_image_path(
                row.frame_path
            )

            # Reuse the decoded frame when adjacent rows reference
            # the same image.
            if image_path != previous_image_path:
                previous_pil_image = load_pil_image(
                    image_path
                )

                previous_image_path = image_path

            if previous_pil_image is None:
                raise RuntimeError(
                    f"No image was loaded at row "
                    f"{position}."
                )

            bbox = normalize_bbox(
                bbox=row.bbox,
                row_number=position,
            )

            label = str(
                row.label
            ).strip()

            label_id = get_label_id(
                label=label,
                row_number=position,
            )

            should_update = (
                position == 1
                or position % PROGRESS_INTERVAL == 0
                or position == rows_to_process
            )

            if should_update:
                print_progress(
                    position=position,
                    total_rows=rows_to_process,
                    image_path=image_path,
                    video_id=str(row.video_id),
                    person_id=str(row.person_id),
                    start_time=start_time,
                )

            yield {
                "source_dataset": str(
                    row.source_dataset
                ),
                "video_id": str(
                    row.video_id
                ),
                "person_id": str(
                    row.person_id
                ),
                "frame_id": int(
                    row.frame_id
                ),

                # Real PIL image. HFImage handles encoding.
                "image": previous_pil_image,

                "bbox": bbox,

                "label": label,
                "label_ID": label_id,
            }

        if (
            MAX_ROWS is not None
            and position >= MAX_ROWS
        ):
            break

    print()

    elapsed = (
        time.perf_counter() - start_time
    )

    print(
        f"Generated {position:,} rows in "
        f"{format_duration(elapsed)}."
    )


# ============================================================
# In-memory validation
# ============================================================

def validate_dataset_before_writing(
    dataset: Dataset,
) -> None:
    """
    Confirm that the generated Dataset has:

    1. a Hugging Face Image feature;
    2. an Arrow image struct;
    3. images that decode into PIL objects;
    4. a valid integer label_ID column.
    """

    if len(dataset) == 0:
        raise ValueError(
            "The generated dataset is empty."
        )

    print("\nDataset features:")
    print(dataset.features)

    image_feature = (
        dataset.features["image"]
    )

    if not isinstance(
        image_feature,
        HFImage,
    ):
        raise TypeError(
            "\nThe image column is not a "
            "Hugging Face Image feature.\n"
            f"Actual feature: {image_feature}"
        )

    if "label_ID" not in dataset.features:
        raise ValueError(
            "The generated dataset has no "
            "label_ID column."
        )

    label_id_feature = (
        dataset.features["label_ID"]
    )

    if not isinstance(
        label_id_feature,
        Value,
    ):
        raise TypeError(
            "label_ID is not a scalar Value feature.\n"
            f"Actual feature: {label_id_feature}"
        )

    if label_id_feature.dtype != "int64":
        raise TypeError(
            "label_ID is not int64.\n"
            f"Actual dtype: {label_id_feature.dtype}"
        )

    arrow_schema = (
        dataset.data.schema
    )

    print("\nArrow schema before writing:")
    print(arrow_schema)

    image_field = (
        arrow_schema.field("image")
    )

    if not pa.types.is_struct(
        image_field.type
    ):
        raise TypeError(
            "\nThe image column is not stored "
            "as an Arrow struct.\n"
            f"Actual type: {image_field.type}"
        )

    child_field_names = {
        field.name
        for field in image_field.type
    }

    if not {
        "bytes",
        "path",
    }.issubset(child_field_names):
        raise TypeError(
            "\nThe image struct must contain "
            "'bytes' and 'path'.\n"
            f"Actual type: {image_field.type}"
        )

    print("\nTesting image decoding...")

    example = dataset[0]
    example_image = example["image"]

    if not isinstance(
        example_image,
        PILImage.Image,
    ):
        raise TypeError(
            "\nThe image did not decode into "
            "a PIL image.\n"
            f"Actual Python type: "
            f"{type(example_image)}"
        )

    print(
        f"Python type: {type(example_image)}"
    )
    print(
        f"Image mode: {example_image.mode}"
    )
    print(
        f"Image size: {example_image.size}"
    )

    print("\nTesting label mapping...")

    example_label = example["label"]
    example_label_id = example["label_ID"]

    expected_label_id = LABEL_TO_ID.get(
        example_label
    )

    if example_label_id != expected_label_id:
        raise ValueError(
            "Incorrect label mapping in first row.\n"
            f"Label: {example_label}\n"
            f"Expected ID: {expected_label_id}\n"
            f"Actual ID: {example_label_id}"
        )

    print(
        f"Label: {example_label}"
    )
    print(
        f"label_ID: {example_label_id}"
    )

    print(
        "\nIn-memory validation successful."
    )


# ============================================================
# Written Parquet validation
# ============================================================

def validate_written_parquet(
    parquet_path: Path,
) -> None:
    """
    Validate the physical Parquet schema and Hugging Face metadata.
    """

    print(
        "\nValidating written Parquet..."
    )

    schema = pq.read_schema(
        str(parquet_path),
        memory_map=False,
    )

    print("\nPhysical Parquet schema:")
    print(schema)

    required_output_columns = {
        "source_dataset",
        "video_id",
        "person_id",
        "frame_id",
        "image",
        "bbox",
        "label",
        "label_ID",
    }

    missing_output_columns = (
        required_output_columns
        - set(schema.names)
    )

    if missing_output_columns:
        raise ValueError(
            "The output Parquet is missing columns: "
            f"{sorted(missing_output_columns)}"
        )

    label_id_field = schema.field(
        "label_ID"
    )

    if not pa.types.is_int64(
        label_id_field.type
    ):
        raise TypeError(
            "label_ID is not stored as int64.\n"
            f"Actual type: {label_id_field.type}"
        )

    image_field = schema.field(
        "image"
    )

    if not pa.types.is_struct(
        image_field.type
    ):
        raise TypeError(
            "\nIncorrect physical image type.\n"
            "Expected a struct containing bytes "
            "and path.\n"
            f"Actual type: {image_field.type}"
        )

    struct_fields = {
        field.name: field.type
        for field in image_field.type
    }

    if "bytes" not in struct_fields:
        raise TypeError(
            "The image struct has no bytes field."
        )

    if "path" not in struct_fields:
        raise TypeError(
            "The image struct has no path field."
        )

    bytes_type = struct_fields["bytes"]
    path_type = struct_fields["path"]

    if not (
        pa.types.is_binary(bytes_type)
        or pa.types.is_large_binary(bytes_type)
    ):
        raise TypeError(
            "image.bytes is not binary.\n"
            f"Actual type: {bytes_type}"
        )

    if not pa.types.is_string(path_type):
        raise TypeError(
            "image.path is not a string.\n"
            f"Actual type: {path_type}"
        )

    metadata = schema.metadata or {}

    if b"huggingface" not in metadata:
        raise ValueError(
            "\nThe Parquet file has no "
            "Hugging Face metadata."
        )

    try:
        hf_metadata = json.loads(
            metadata[b"huggingface"].decode(
                "utf-8"
            )
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            "The Hugging Face metadata could "
            "not be decoded."
        ) from error

    features_metadata = (
        hf_metadata
        .get("info", {})
        .get("features", {})
    )

    image_metadata = (
        features_metadata.get("image")
    )

    label_id_metadata = (
        features_metadata.get("label_ID")
    )

    if image_metadata is None:
        raise ValueError(
            "Hugging Face metadata does not "
            "describe the image column."
        )

    if image_metadata.get("_type") != "Image":
        raise TypeError(
            "\nThe metadata does not declare "
            "image as a Hugging Face Image.\n"
            f"Actual metadata: {image_metadata}"
        )

    if label_id_metadata is None:
        raise ValueError(
            "Hugging Face metadata does not "
            "describe the label_ID column."
        )

    if (
        label_id_metadata.get("dtype")
        != "int64"
    ):
        raise TypeError(
            "Hugging Face metadata does not declare "
            "label_ID as int64.\n"
            f"Actual metadata: {label_id_metadata}"
        )

    print("\nHugging Face image metadata:")
    print(image_metadata)

    print("\nHugging Face label_ID metadata:")
    print(label_id_metadata)

    print(
        "\nWritten Parquet validation successful."
    )


# ============================================================
# Conversion
# ============================================================

def convert_parquet(
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Convert one input Parquet file.
    """

    print(
        f"\nReading metadata: {input_path}"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input Parquet does not exist: "
            f"{input_path}"
        )

    (
        complete_row_count,
        rows_to_process,
    ) = validate_input_columns(
        input_path
    )

    print(
        f"Complete input rows: "
        f"{complete_row_count:,}"
    )
    print(
        f"Rows to process: "
        f"{rows_to_process:,}"
    )

    file_cache_directory = (
        CACHE_DIRECTORY
        / input_path.stem
    )

    if FORCE_REGENERATE_CACHE and file_cache_directory.exists():
        print(
            "FORCE_REGENERATE_CACHE=True; removing existing cache: "
            f"{file_cache_directory}"
        )
        shutil.rmtree(file_cache_directory)

    file_cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if FORCE_REGENERATE_CACHE:
        print("Generator cache reuse is disabled for this run.")
    else:
        print(
            "Generator cache reuse is enabled. If the previous generation "
            "completed and its fingerprint matches, Hugging Face will load "
            "the cached Arrow dataset instead of decoding all images again."
        )
        print(f"Cache directory: {file_cache_directory}")

    print(
        "Creating Hugging Face Dataset..."
    )

    dataset = Dataset.from_generator(
        generate_rows,
        gen_kwargs={
            "input_path_string": str(
                input_path
            ),
        },
        features=FEATURES,
        keep_in_memory=False,
        cache_dir=str(
            file_cache_directory
        ),
    )

    # Do not call dataset.cast(FEATURES) here.
    # FEATURES was already supplied to Dataset.from_generator(), so the
    # generated dataset already has the intended schema. Casting the complete
    # image dataset forces Arrow to combine very large binary chunks and can
    # raise: ArrowInvalid: offset overflow while concatenating arrays.

    if len(dataset) != rows_to_process:
        raise ValueError(
            f"Expected {rows_to_process:,} generated "
            f"rows, but found {len(dataset):,}."
        )

    validate_dataset_before_writing(
        dataset
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        print(
            f"\nRemoving previous output: "
            f"{output_path}"
        )

        output_path.unlink()

    print(
        f"\nWriting final Parquet: "
        f"{output_path}"
    )

    write_start_time = (
        time.perf_counter()
    )

    dataset.to_parquet(
        str(output_path),
        batch_size=PARQUET_WRITE_BATCH_SIZE,
    )

    write_duration = (
        time.perf_counter()
        - write_start_time
    )

    validate_written_parquet(
        output_path
    )

    output_size_mb = (
        output_path.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"\nFinished writing "
        f"{output_path.name}."
    )
    print(
        f"Rows written: {len(dataset):,}"
    )
    print(
        f"Output size: "
        f"{output_size_mb:.2f} MB"
    )
    print(
        "Writing time: "
        f"{format_duration(write_duration)}"
    )

    del dataset

    if KEEP_GENERATOR_CACHE:
        print(
            "Keeping generator cache for possible reuse: "
            f"{file_cache_directory}"
        )
    elif file_cache_directory.exists():
        print("Removing generator cache after successful completion...")
        shutil.rmtree(file_cache_directory)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Convert every file listed in PARQUET_FILES.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLabel mapping:")

    for label, label_id in LABEL_TO_ID.items():
        print(
            f"  {label} -> {label_id}"
        )

    if MAX_ROWS is None:
        print(
            "\nMAX_ROWS = None"
            "\nEvery row will be processed."
        )

    else:
        print(
            f"\nMAX_ROWS = {MAX_ROWS:,}"
            f"\nOnly the first {MAX_ROWS:,} rows "
            "of each file will be processed."
        )

    overall_start_time = (
        time.perf_counter()
    )

    for file_number, input_path in enumerate(
        PARQUET_FILES,
        start=1,
    ):
        print(
            f"\n{'=' * 80}\n"
            f"File {file_number}/"
            f"{len(PARQUET_FILES)}: "
            f"{input_path.name}\n"
            f"{'=' * 80}"
        )

        output_path = get_output_path(
            input_path
        )

        convert_parquet(
            input_path=input_path,
            output_path=output_path,
        )

    overall_duration = (
        time.perf_counter()
        - overall_start_time
    )

    print(
        f"\nAll Parquet files converted in "
        f"{format_duration(overall_duration)}."
    )

    if (
        CACHE_DIRECTORY.exists()
        and not any(
            CACHE_DIRECTORY.iterdir()
        )
    ):
        CACHE_DIRECTORY.rmdir()


if __name__ == "__main__":
    main()