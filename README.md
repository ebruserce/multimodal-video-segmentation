# Gaze Analysis Pipeline

A research pipeline for segmenting stimulus videos and aligning gaze data across participants. Developed for studying differences in gaze behavior between children with and without autism (ASD, NT, ATP groups).

---

## Project Overview

Children watch stimulus videos (two people facing each other, performing an activity) while their gaze is tracked. This pipeline:

1. Segments each stimulus video into labeled regions (head, arms, hands, body, activity)
2. Computes how much each video's people positions differ from a reference video
3. Shifts each participant's gaze data to a common coordinate space
4. Generates heatmaps and gaze visualizations overlaid on a reference image, split by diagnosis group

---

## Repository Structure

```
project/
├── configs/
│   ├── base.yaml               # Main config (paths, settings, reference video name)
│   └── paths.local.yaml        # Local path overrides (not committed)
├── data/
│   ├── input/           # Raw stimulus .mp4 files
│   ├── output/                 # Segmented images + reference image
│   └── landmarks/              # Head center CSVs + shifts CSV
├── models/                     # MediaPipe model files
├── segment_frames.py           # Script 1: segment first frame of each video
└── compute_shifts.py           # Script 2: compute per-video alignment shifts
```

---

## Scripts

### Script 1: `segment_frames.py`

Processes the first frame of each input video. For each video, produces:

- A segmented image (`output/<video_stem>_segmented.jpg`) with colored region overlays:
  - **Red** — head
  - **Green** — arms
  - **Dark green** — hands
  - **Blue** — body
  - **Yellow** — activity region (between the two people)
- A single CSV (`landmarks/head_centers.csv`) with one row per video containing the left and right person's head center coordinates

**Color coding is for visual inspection only.** The segmented image of the first video listed is used as the reference image for final visualizations.

**To run:**
```bash
python segment_frames.py
```

---

### Script 2: `compute_shifts.py`

Reads `landmarks/head_centers.csv` and computes how much each video's head positions need to shift to match the reference video.

Produces:
- `landmarks/shifts.csv` — one row per video with `left_shift_x`, `left_shift_y`, `right_shift_x`, `right_shift_y`

The reference video is set in `base.yaml` under `settings.reference_video` (use the input video filename stem, no extension).

Shift logic:
- If a gaze point's x coordinate is less than `width / 2`, apply the left shift
- Otherwise, apply the right shift

**To run:**
```bash
python compute_shifts.py
```

---

## Configuration

In `base.yaml`, set:

```yaml
settings:
  reference_video: "video_stem_here"   # filename without extension
  alpha: 0.5                           # overlay transparency
  num_poses: 2
  segmentation_threshold: 0.5
  mask_dilation_kernel: 15
  min_radius: 30
  video_input_extension: "*.mp4"
  video_output_extension: "mp4v"

paths:
  project_root: "."
  models:
    pose_landmarker: "models/pose_landmarker.task"
  data:
    input_videos: "data/input_videos"
    output: "data/output"
    landmarks: "data/landmarks"
```

---

## What Is Done

- [x] Pose-based segmentation of people into head, arms, hands, body regions
- [x] Activity region derived from pose landmarks (no object detection model needed)
- [x] Script processes all input videos, outputs segmented first frame per video
- [x] Head centers saved to CSV for alignment
- [x] Shift computation relative to a chosen reference video

---

## What Is Left To Do

These steps are remaining before the poster deadline:

### Ebru
- [ ] Verify segmentation output looks correct across all 5 videos
- [ ] Confirm reference video is set correctly in `base.yaml`
- [ ] Investigate height normalization across videos (people may sit at different heights/distances)

### Brenda
- [ ] Obtain gaze data format and confirm column names (frame, x, y, participant ID, diagnosis group)
- [ ] Write `apply_shifts.py` (Script 3):
  - Load `shifts.csv`
  - For each gaze point: if `x < width/2` apply left shift, else apply right shift
  - Output shifted gaze coordinates per video
- [ ] Write `visualize.py` (Script 4):
  - Split shifted gaze data by group (ASD, NT, ATP)
  - Generate heatmap per group
  - Overlay heatmap onto reference segmented image from Script 1
  - Optional: add per-participant gaze traces that fade over time

### Both
- [ ] Validate alignment visually — plot shifted gaze points on reference image and check they land in plausible regions
- [ ] Compute % gaze on head, % gaze on activity, % gaze elsewhere per group
- [ ] Produce side-by-side comparison figures for poster

---

## Known Limitations

- Segmentation is based on the first frame only — does not account for movement across the video (intentional simplification for now)
- Activity region mask is a rectangle derived from elbow and foot landmarks — not a precise object mask
- Alignment is translation-only (no rotation or scaling) — gaze points near the center seam (`x ≈ width/2`) may have slightly higher error, but this region is less critical for the head vs. activity analysis
- Pipeline currently only handles the AM stimulus type (two people facing each other). Robot stimuli are not supported yet.
