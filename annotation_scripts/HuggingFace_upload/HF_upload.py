from pathlib import Path

from huggingface_hub import HfApi


# ============================================================
# Configuration
# ============================================================

UPLOAD_DIRECTORY = Path(
    r"D:\EngagementDetection\Push_to_HF\parquet_final_public"
)

REPO_ID = "marilynchahine1/interaction-readiness-detection-dataset-frame-path-version"

REQUIRED_FILES = [
    "README.md",
    "AVIDAR.parquet",
    "JRDB.parquet",
    "SSUP-HRI.parquet",
    "JSON.zip",
    "frames.zip"
]


# ============================================================
# Validation
# ============================================================

def validate_upload_directory() -> None:
    if not UPLOAD_DIRECTORY.is_dir():
        raise NotADirectoryError(
            f"Upload directory does not exist: {UPLOAD_DIRECTORY}"
        )

    missing_files = []

    for filename in REQUIRED_FILES:
        file_path = UPLOAD_DIRECTORY / filename

        if not file_path.is_file():
            missing_files.append(file_path)

    if missing_files:
        formatted = "\n".join(
            f"  - {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "The following required files are missing:\n"
            f"{formatted}"
        )

    readme_path = UPLOAD_DIRECTORY / "README.md"

    if not readme_path.read_text(
        encoding="utf-8"
    ).strip():
        raise ValueError(
            f"README.md is empty: {readme_path}"
        )


# ============================================================
# Upload
# ============================================================

def main() -> None:
    validate_upload_directory()

    api = HfApi()

    print(
        f"Creating or checking repository "
        f"'{REPO_ID}'..."
    )

    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    print("Repository ready.")
    print(f"Uploading folder: {UPLOAD_DIRECTORY}")
    print(
        "The repository will remain private. "
        "Large uploads may take several hours."
    )

    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(UPLOAD_DIRECTORY),
    )

    print("\nUpload complete.")
    print(
        "Private dataset repository:\n"
        f"https://huggingface.co/datasets/{REPO_ID}"
    )


if __name__ == "__main__":
    main()
