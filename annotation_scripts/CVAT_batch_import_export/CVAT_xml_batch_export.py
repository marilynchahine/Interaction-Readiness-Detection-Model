from pathlib import Path
import subprocess
import json
import sys
import zipfile
import shutil

# ==========================
# CONFIGURATION
# ==========================

CVAT_HOST = "http://localhost:8080"
USERNAME = "marilynchahine"
PASSWORD = "CVAT170801!"

OUTPUT_DIR = Path(r"C:\Users\maril\OneDrive\Desktop\GitHub\InteractionReadiness\IRDM\data\CVAT_files_labels\JRDB")

# For video tasks, this is usually correct:
EXPORT_FORMAT = "CVAT for video 1.1"

# Set to None to export all tasks
# Or put specific task IDs, e.g. [12, 13, 14]
TASK_IDS_TO_EXPORT = list(range(56, 191))

# ==========================

print("Exporting annotations from CVAT.")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CVAT_CMD = [
    sys.executable,
    "-m",
    "cvat_cli",
    "--server-host",
    CVAT_HOST,
    "--auth",
    f"{USERNAME}:{PASSWORD}",
]


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("\nCommand failed:")
        print(" ".join(cmd))
        print(result.stderr)
        raise RuntimeError("Command failed")

    return result.stdout


print("Retrieving CVAT task list...")

tasks_json = run_cmd(CVAT_CMD + ["task", "ls", "--json"])
tasks = json.loads(tasks_json)

if TASK_IDS_TO_EXPORT is not None:
    tasks = [task for task in tasks if task["id"] in TASK_IDS_TO_EXPORT]

print(f"Found {len(tasks)} task(s) to export.")

for task in tasks:
    task_id = task["id"]
    task_name = Path(task["name"]).stem

    output_path = OUTPUT_DIR / f"{task_name}.zip"

    print(f"Exporting task {task_id}: {task['name']} -> {output_path.name}")

    run_cmd(
        CVAT_CMD
        + [
            "task",
            "export-dataset",
            "--format",
            EXPORT_FORMAT,
            str(task_id),
            str(output_path),
        ]
    )

print("Done.")


print("Extracting and renaming zip files.")

ZIP_FOLDER = Path(r"C:\Users\maril\OneDrive\Desktop\GitHub\InteractionReadiness\IRDM\data\CVAT_files_labels\JRDB")
OUTPUT_FOLDER = Path(r"C:\Users\maril\OneDrive\Desktop\GitHub\InteractionReadiness\IRDM\data\CVAT_files_labels\JRDB")

OUTPUT_FOLDER.mkdir(exist_ok=True)

for zip_path in ZIP_FOLDER.glob("*.zip"):
    with zipfile.ZipFile(zip_path) as z:
        xml_name = next(f for f in z.namelist() if f.endswith(".xml"))

        temp_path = OUTPUT_FOLDER / xml_name
        z.extract(xml_name, OUTPUT_FOLDER)

        shutil.move(
            temp_path,
            OUTPUT_FOLDER / f"{zip_path.stem}.xml"
        )

print("Done!")