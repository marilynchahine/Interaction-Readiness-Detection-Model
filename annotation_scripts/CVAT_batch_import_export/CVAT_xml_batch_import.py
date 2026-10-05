from pathlib import Path
import subprocess
import json
import sys

# ==========================
# CONFIGURATION
# ==========================

CVAT_HOST = "http://localhost:8080"   # e.g. http://localhost:8080
USERNAME = "marilynchahine"
PASSWORD = "CVAT170801!"

XML_DIR = Path(r"D:\EngagementDetection\data\CVAT_xml_processed\SSUP-HRI\AstorPlace_final")

# Try "CVAT 1.1" first.
# If CVAT says the format is unknown, change it to:
# "CVAT for video 1.1"
ANNOTATION_FORMAT = "CVAT 1.1"

# ==========================


def run_cmd(cmd):
    """Run a command and return stdout."""
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
        raise RuntimeError("Command failed.")

    return result.stdout


# Use the same Python interpreter that's running this script
CVAT_CMD = [
    sys.executable,
    "-m",
    "cvat_cli",
    "--server-host",
    CVAT_HOST,
    "--auth",
    f"{USERNAME}:{PASSWORD}",
]

print("Retrieving task list from CVAT...")

tasks_json = run_cmd(
    CVAT_CMD
    + [
        "task",
        "ls",
        "--json",
    ]
)

tasks = json.loads(tasks_json)

print(f"Found {len(tasks)} tasks.")

# Build dictionary:
# JRDB_001 -> task ID
task_map = {}

for task in tasks:
    task_name = Path(task["name"]).stem
    task_map[task_name] = task["id"]

print(f"Found {len(list(XML_DIR.glob('*.xml')))} XML files.\n")

uploaded = 0
missing = 0

for xml_path in sorted(XML_DIR.glob("*.xml")):

    base_name = xml_path.stem

    if base_name not in task_map:
        print(f"❌ No matching task for {base_name}")
        missing += 1
        continue

    task_id = task_map[base_name]

    print(f"Uploading {xml_path.name} -> Task {task_id}")

    run_cmd(
        CVAT_CMD
        + [
            "task",
            "import-dataset",
            "--format",
            ANNOTATION_FORMAT,
            str(task_id),
            str(xml_path),
        ]
    )

    uploaded += 1

print("\n==========================")
print(f"Uploaded : {uploaded}")
print(f"Missing  : {missing}")
print("==========================")
print("Done!")