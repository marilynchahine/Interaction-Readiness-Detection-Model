# Data Annotation Steps

Important: check dataset-specific instructions for variations or additional steps done per dataset

1 - download raw video format of datasets and place them in data/rawVideo/dataset_name

2 - run annotation_scripts/AnnotationPipeline.py to generate the videos' bounding boxes and tracks per person
	NB: don't forget to specify the input path (data/rawVideo/dataset_name) and output path 	(data/CVAT_files_bboxes/dataset_name)

For next steps, check CVAT_instructions.txt for instructions on how to launch CVAT:

3 - CVAT: Create a project, and create a label within it with attribute 'interaction_readiness' that takes 'not_interaction_ready', 'interaction_ready', and 'interaction_ongoing' as values

4 - CVAT: Import videos, and their corresponding bounding boxes/tracks XML files, make sure all the bounding boxes are valid 
	(use scripts in CVAT_batch_import_export to do these steps in batches)

5 - For each dataset, create a spreadsheet describing the ranges of labels per person (ex: frames 1 to 100 is interaction_ready) and the manual corrections to do outside CVAT

6 - Run CVATXML_to_JSON.py to convert the CVAT-generated annotations to the desired JSON format
	NB: don't forget to specify the input path (data/CVAT_files_labels/dataset_name) and output path (data/JSON_final/dataset_name)


CVATXML_to_JSON.py running command in bash:
python CVAT_to_JSON.py --xml-dir "[Replace with XML to convert directory]" --output-dir "[Replace with output directory for JSON files]"



## Launching CVAT Instructions

1 - Open Docker Desktop
2 - Open CVAT directory in CMD
3 - Run 'docker compose up -d'
4 - Open http://localhost:8080 (or your CVAT_HOST) in your browser.
5 - Log in with your superuser account.
6 - Create a project or task, upload your data (images, videos, or point clouds), and define labels to start annotating.



## AVIDAR-Specific Instructions

ultralytics bounding boxes and tracking generated tracks that marged into one another or generated phantom boxes when participants crossed each other (one person passes in front of the other).

Solution: 
AVIDAR_ManualCorrections.txt contains all the corrections that have been done manually on the final JSON files to keep consistent person ID despite ultralytics mix-ups



## JRDB-Specific Instructions

JRDB only provides its data in image format, JRDB_specific/frames_to_videos.py was used to convert the frames into videos. Frames corresponding to one video are placed in the same folder, the script takes a directory containing multiple folders and converts the content of each folder into a video.

JRDB already contained skeleton data, ultralytics was skipped and the bounding boxes were generated using JRDB_specific/skeletons_to_bboxes.py to convert skeletons to bounding boxes by taking the minimal rectangle that contains all joints as a bounding box for each person for each frame.



## Other Information/Scripts

data_stats.py calculates statistics related to a specified dataset:
- Total videos
- Total people
- Total person-frame annotations
- Average frames per person
- Frames per label
- People per label
