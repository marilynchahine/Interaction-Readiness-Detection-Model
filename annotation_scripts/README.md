# Annotation Scripts

This folder contains the scripts used to build the interaction-readiness dataset from AVDIAR, JRDB, and SSUP-HRI.

The main workflow is:

```text
raw data
  ↓
person detection / tracking
  ↓
CVAT annotation
  ↓
CVAT export
  ↓
JSON conversion
  ↓
dataset balancing
  ↓
frame-based and temporal formats
```

> Dataset-specific differences are described below.

---

## Main Annotation Pipeline

### 1. Prepare the source videos

Download the original datasets and place the videos in your local data directory.

For JRDB, which is provided as image sequences, first use:

```text
JRDB_specific/frames_to_video.py
```

---

### 2. Generate person tracks

For AVDIAR and SSUP-HRI, run:

```bash
python annotation_scripts/ultralytics_Pipeline.py
```

This script:

- detects people with Ultralytics YOLO;
- tracks them with BoT-SORT;
- writes CVAT-compatible XML person tracks.

Update:

```python
INPUT_DIR
OUTPUT_DIR
MODEL_PATH
```

before running.

JRDB uses the provided skeleton annotations instead of YOLO. See the JRDB section below.

---

### 3. Annotate in CVAT

Launch CVAT locally:

```bash
docker compose up -d
```

Then open:

```text
http://localhost:8080
```

Create a `person` label with the attribute:

```text
interaction_readiness
```

with values:

```text
not_interaction_ready
interaction_ready
interaction_ongoing
```

Import the videos and person-track XML files.

For batch import, use:

```text
CVAT_batch_import_export/CVAT_xml_batch_import.py
```

After annotation, export the tasks with:

```text
CVAT_batch_import_export/CVAT_xml_batch_export.py
```

---

### 4. Convert CVAT annotations to JSON

Run:

```bash
python annotation_scripts/file_types_conversions/CVAT_to_JSON.py
```

This converts CVAT XML tracks to the hierarchical JSON format:

```json
{
  "video_id": "example_video",
  "person_0": [
    {
      "frame_id": 0,
      "bbox": [x, y, width, height],
      "label": "not_interaction_ready"
    }
  ]
}
```

---

### 5. Apply dataset-specific corrections

For AVDIAR, tracking errors caused by identity switches and crossings were manually corrected.

The applied corrections are listed in:

```text
AVIDAR_ManualCorrections.txt
```

---

### 6. Extract frames

Run:

```bash
python annotation_scripts/file_types_conversions/video_to_frames.py
```

This extracts all frames to:

```text
<output>/
└── <video_name>/
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
```

---

### 7. Balance the dataset

For the frame-based dataset, use:

```bash
python annotation_scripts/data_balancing.py
```

For the final temporal dataset, use:

```bash
python annotation_scripts/data_balancing_video.py
```

The balancing scripts:

- retain the main interaction-ready / interaction-ongoing examples;
- add negative examples from AVDIAR and JRDB;
- remove frames containing only `interaction_done`;
- limit the selected people to at most five.

For the temporal dataset, person selection is performed consistently at the video level.

---

### 8. Add Train / Val / Test splits

The balanced CSV must contain a:

```text
split
```

column.

Splits should be assigned at the **video level** to prevent neighboring frames from the same video from appearing in different subsets.

---

### 9. Create 16-frame temporal groups

Run:

```bash
python annotation_scripts/group_by_16frames.py
```

This converts the balanced temporal CSV into 16-frame sliding windows:

```text
frames 1–16
frames 2–17
frames 3–18
...
```

The resulting JSON files are used by the video-based model scripts.

---

## Script Reference

| Script | Purpose |
|---|---|
| `ultralytics_Pipeline.py` | Person detection and tracking; exports CVAT XML. |
| `data_balancing.py` | Builds the balanced frame-based dataset and an earlier temporal version. |
| `data_balancing_video.py` | Builds the final temporal dataset with consistent person selection across videos. |
| `dataset_stats.py` | Computes dataset, video, person, and label statistics. |
| `group_by_16frames.py` | Converts the temporal CSV into 16-frame sliding-window JSON files. |
| `CVAT_batch_import_export/CVAT_xml_batch_import.py` | Batch-imports XML annotations into CVAT tasks. |
| `CVAT_batch_import_export/CVAT_xml_batch_export.py` | Batch-exports CVAT annotations. |
| `file_types_conversions/CVAT_to_JSON.py` | Converts CVAT XML to hierarchical JSON. |
| `file_types_conversions/JSON_to_CVAT.py` | Converts hierarchical JSON back to CVAT XML. |
| `file_types_conversions/JSON_to_Parquet.py` | Converts hierarchical JSON to Parquet. |
| `file_types_conversions/video_to_frames.py` | Extracts video frames. |
| `file_types_conversions/frames_to_cropped_frames_per_person.py` | Creates person crops from annotated frames. |
| `JRDB_specific/frames_to_video.py` | Converts JRDB image sequences into videos for CVAT. |
| `JRDB_specific/skeletons_to_bboxes.py` | Converts JRDB skeleton annotations into person bounding boxes. |
| `HuggingFace_upload/selfContained_parquet.py` | Creates self-contained Parquet files with embedded images. |
| `HuggingFace_upload/HF_upload.py` | Uploads the prepared dataset to Hugging Face. |

---

## JRDB-Specific Workflow

JRDB already provides skeleton annotations, so YOLO tracking is not used.

Use:

```text
JRDB frames
    ↓
JRDB_specific/frames_to_video.py
    ↓
videos for CVAT
```

and:

```text
JRDB skeleton annotations
    ↓
JRDB_specific/skeletons_to_bboxes.py
    ↓
CVAT XML person tracks
```

Then continue with the normal CVAT annotation and JSON conversion workflow.

---

## Dataset Statistics

To inspect the hierarchical JSON dataset, run:

```bash
python annotation_scripts/dataset_stats.py
```

The script reports statistics such as:

- number of videos;
- number of people;
- person-frame annotations;
- frames per label;
- people per label;
- per-video statistics.

---

## Hugging Face Export

For publication, the dataset can be converted to Parquet (used in a previous version) and uploaded with:

```text
HuggingFace_upload/selfContained_parquet.py
HuggingFace_upload/HF_upload.py
```

The released dataset is available at:

https://huggingface.co/datasets/marilynchahine1/interaction-readiness-detection-dataset-frame-path-version

---

## Configuration

Most scripts use paths defined directly inside the file.

Before running, update variables such as:

```python
INPUT_DIR
OUTPUT_DIR
JSON_DIR
FRAME_ROOT
VIDEO_DIR
XML_DIR
```

The committed paths are machine-specific and should be replaced with local paths.
