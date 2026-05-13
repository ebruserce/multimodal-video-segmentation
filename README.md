# Multimodal Video Segmentation Pipeline

## Overview

This project implements a video segmentation and gaze classification pipeline for social attention analysis. The pipeline takes AM stimulus videos and eye-tracking gaze data as input, then automatically classifies where participants are looking over time.

The current workflow consists of:

1. Extracting pose landmarks and segmentation masks using MediaPipe
2. Generating semantic segmentation regions for each video
3. Classifying gaze points into predicted AOIs
4. Validating predicted AOIs against the original gaze labels
5. Attaching participant metadata for later group-level analysis

The pipeline is currently designed for AM stimulus videos with two interacting actors. The regions of interest include:

- Head
- Hands
- Arms
- Body
- Activity region between the two actors
- Background
- Screen border

The immediate goal is to compare predicted AOIs against the original AOI labels in the gaze data and identify segmentation settings, especially head dilation, that produce the most accurate mappings.

---

# Pipeline Structure

| Script | Purpose |
|---|---|
| `pose_coordinates.py` | Extracts MediaPipe pose landmarks and segmentation masks |
| `segmentations.py` | Generates semantic segmentation regions |
| `gaze_locations.py` | Classifies gaze points into AOIs |
| `validate.py` | Evaluates AOI prediction accuracy across head dilation levels |
| `attach_participant_metadata.py` | Adds participant diagnosis and age metadata |
| `main.py` | Runs the full pipeline |

---

# Setup Instructions

## Python Version

MediaPipe must be run with:

```text
Python 3.12 or lower
```

I have been running this project in a Mamba environment on the HPC.

Example environment setup:

```bash
mamba create -n video-seg python=3.11
mamba activate video-seg
```

## Required Packages

Install the required packages with:

```bash
pip install mediapipe opencv-python pandas numpy pyyaml
```

---

# Folder Structure

The expected project structure is:

```text
project_root/
│
├── configs/
│   ├── base.yaml
│   └── paths.local.yaml
│
├── data/
│   ├── input/
│   │   ├── videos/
│   │   │   ├── AM_A1_S5_B2_GA_D1_F1.avi
│   │   │   └── ...
│   │   │
│   │   ├── gaze/
│   │   │   ├── AM_A1_S5_B2_GA_D1_F1/
│   │   │   │   ├── participant1.csv
│   │   │   │   └── ...
│   │   │   ├── AM_A1_S5_B2_GA_D1_F2/
│   │   │   │   ├── participant1.csv
│   │   │   │   └── ...
│   │   │   └── ...
│   │   │
│   │   └── am_edf_ch_mapping_v01.csv
│   │
│   ├── landmarks/
│   │   └── frames/
│   │
│   └── output/
│
├── models/
│   └── pose_landmarker.task
│
└── scripts/
    └── segmentation/
        ├── pose_coordinates.py
        ├── segmentations.py
        ├── gaze_locations.py
        ├── validate.py
        ├── attach_participant_metadata.py
        └── main.py
```

---

# Input Requirements

## Videos

Stimulus videos should be placed in:

```text
data/input/videos/
```

Video names should match the gaze subfolder names.

Example:

```text
AM_A1_S5_B2_GA_D1_F1.avi
```

## Gaze Data

Gaze data should be organized by stimulus video:

```text
data/input/gaze/[video_name]/
```

Example:

```text
data/input/gaze/AM_A1_S5_B2_GA_D1_F1/
```

Each gaze CSV is expected to include the following columns:

```text
t
sx
sy
valid
aoi
```

where:

- `t` is the gaze timestamp
- `sx` and `sy` are screen-space gaze coordinates
- `valid` indicates whether the gaze point is valid
- `aoi` is the original AOI label used for validation

## Metadata File

Participant metadata should be stored at:

```text
data/input/am_edf_ch_mapping_v01.csv
```

This file is used to attach participant-level information to gaze outputs.

Expected metadata columns include:

```text
edf
dx
et_age_corrected_months
```

Additional columns may also be present and can be preserved in the metadata merge step.

---

# Running the Pipeline

## Full Pipeline

To run the full pipeline:

```bash
python scripts/segmentation/main.py
```

The full pipeline runs the major processing steps in order, depending on the flags set inside `main.py`.

---

# Running Individual Steps

## 1. Pose Coordinate Extraction

```bash
python scripts/segmentation/pose_coordinates.py
```

This script runs MediaPipe on each video and outputs:

- pose landmark CSVs
- MediaPipe segmentation mask `.npy` files

Outputs are saved to:

```text
data/landmarks/frames/
```

### Important

`pose_coordinates.py` only needs to be run once per video.

Once pose coordinates and masks have been generated, they can be reused for segmentation, validation, and dilation testing.

---

## 2. Segmentation Generation

```bash
python scripts/segmentation/segmentations.py
```

This script generates segmentation region definitions based on the pose coordinates.

Outputs are saved to:

```text
data/landmarks/frames/
```

### Important

`segmentations.py` only needs to be run once per video per head dilation level.

---

## 3. Gaze Location Classification

```bash
python scripts/segmentation/gaze_locations.py
```

This script classifies each gaze point into a predicted AOI.

It reads gaze files from:

```text
data/input/gaze/[video_name]/
```

and writes one combined gaze-location CSV per video.

Example output:

```text
data/output/gaze_locations/AM_A1_S5_B2_GA_D1_F1/
└── AM_A1_S5_B2_GA_D1_F1_gaze_locations_all.csv
```

Each row includes:

```text
video_name
participant_id
frame
t
sx
sy
predicted_aoi
```

The `frame` column is included because gaze timestamps are not always aligned across participants. Later group-level visualization can use frame number as the x-axis.

---

## 4. Dilation Validation

```bash
python scripts/segmentation/validate.py
```

This script tests head dilation levels and compares predicted AOIs against the original `aoi` column in the gaze data.

Currently, the script tests head dilation values from:

```text
0.1 to 1.1
```

The goal is to identify the head dilation value that gives the best overall AOI mapping accuracy for each video.

Outputs include:

```text
data/output/dilation_validation/
├── recommended_head_dilations.csv
└── [video_name]/
    └── [video_name]_dilation_validation_summary.csv
```

The per-video summary CSV includes columns such as:

```text
video_name
head_dilation
num_participants
avg_overall_pct
avg_total_head_pct
avg_a1head_pct
avg_a2head_pct
avg_activity_pct
avg_bg_pct
```

The recommendation file contains the selected dilation value for each video.

---

## 5. Attach Participant Metadata

```bash
python scripts/segmentation/attach_participant_metadata.py
```

This script attaches participant metadata to the gaze-location outputs.

It reads:

```text
data/output/gaze_locations/[video_name]/[video_name]_gaze_locations_all.csv
```

and writes:

```text
data/output/gaze_locations_with_metadata/[video_name]/
└── [video_name]_gaze_locations_with_metadata.csv
```

Rows without matching metadata are saved separately:

```text
data/output/gaze_locations_with_metadata/[video_name]/
└── [video_name]_gaze_locations_missing_metadata.csv
```

The metadata-enriched files include columns such as:

```text
dx
diagnosis_group
age_months
age_group
group_label
risk_group
visit
id
```

Diagnosis groups are currently normalized as:

```text
ASD -> ASD
NT  -> NASD
ATP -> NASD
```

Age groups are currently assigned as:

```text
infant  = age < 18 months
toddler = age >= 18 months
```

---

# Segmentation Video Rendering

Currently, segmentation video rendering is commented out in:

```text
segmentations.py
```

To render segmentation visualization videos, locate the `render_segmentation_video(...)` call near the bottom of `segmentations.py` and uncomment it.

Rendering videos can be slow, so this step is disabled by default.

---

# Sharing Without MediaPipe

MediaPipe can be difficult to set up depending on the system environment. To make it easier for collaborators to run the downstream pipeline, one option is to share the already-generated pose outputs instead of requiring them to run MediaPipe.

For a given video, share:

```text
data/landmarks/frames/[video_name]_pose.csv
data/landmarks/frames/[video_name]_masks.npy
```

Then the collaborator can run the remaining scripts without running `pose_coordinates.py`.

This is useful if the collaborator only needs to test segmentation, gaze classification, validation, or group-difference analysis.

---

# Current Limitations and Assumptions

## Body Mapping Bug

There is currently a known bug involving mapping gaze points to body masks. Body-region classification is therefore not fully reliable at the moment.

The current analysis mainly focuses on head, activity, and background regions.

## Head Dilation Only

Currently, the only dilation parameter being tuned is:

```text
head dilation
```

Other segmentation parameters are held fixed.

## Activity Region Approximation

The activity region is currently estimated as the area between the two actors. It is computed once from the first detected frame and remains fixed throughout the video.

This assumes that the shared activity space does not substantially change over time.

## Actor Assignment

Actors are currently assigned based on left/right ordering in the frame.

This works for the current videos but may fail if MediaPipe swaps detections between actors.

## Coordinate Assumptions

The pipeline assumes that gaze coordinates align with a 1680 × 1050 canvas. Videos are centered inside this canvas, and no video resizing is currently performed.

## MediaPipe Dependency

Only `pose_coordinates.py` requires MediaPipe. If pose CSVs and mask files are already available, the later scripts can be run without rerunning MediaPipe.

---

# Outputs

## Pose and Segmentation Outputs

```text
data/landmarks/frames/
```

Contains:

```text
[video_name]_pose.csv
[video_name]_masks.npy
[video_name]_segmentations.csv
```

## Gaze Classification Outputs

```text
data/output/gaze_locations/
```

Contains one combined gaze-location file per video.

## Metadata-Enriched Outputs

```text
data/output/gaze_locations_with_metadata/
```

Contains gaze-location files with participant metadata attached.

## Validation Outputs

```text
data/output/dilation_validation/
```

Contains dilation summaries and recommended head dilation values.

---

# Future Work

Planned next steps include:

- Fixing body-mask gaze classification
- Improving actor tracking across frames
- Testing additional segmentation parameters beyond head dilation
- Exploring dynamic activity regions
- Generating group-level AOI timecourse visualizations
- Comparing gaze patterns between ASD and NASD groups
- Comparing gaze patterns between infants and toddlers
- Building classification models using gaze-derived features

---

# Notes

This code is currently under active development. The current focus is validating the segmentation and gaze-mapping pipeline before moving into group-difference analysis and classification modeling.