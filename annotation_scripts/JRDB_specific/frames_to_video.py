from pathlib import Path
import subprocess

# Change these paths
input_dir = Path("C:\\Users\\maril\\OneDrive\\Desktop\\GitHub\\InteractionReadiness\\IRDM\\data\\rawVideo\\JRDB\\framesPerVideo\\stitched")
output_dir = Path("C:\\Users\\maril\\OneDrive\\Desktop\\GitHub\\InteractionReadiness\\IRDM\\data\\rawVideo\\JRDB\\videos\\stitched")

fps = 30  # use 15 for JRDB 360 RGB, 30 for RGB-D

output_dir.mkdir(parents=True, exist_ok=True)

for folder in input_dir.iterdir():
    if not folder.is_dir():
        continue

    # Output video has the same name as the folder
    output_video = output_dir / f"{folder.name}.mp4"

    # Assumes frames are named 000001.jpg, 000002.jpg, ...
    input_pattern = folder / "%06d.jpg"

    command = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(input_pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_video)
    ]

    print(f"Converting {folder.name} -> {output_video.name}")
    subprocess.run(command, check=True)

print("Done.")