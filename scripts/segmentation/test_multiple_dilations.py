# Ebru Serce, 2026
import cv2
import re
import yaml
import numpy as np
from pathlib import Path
import csv

from pose_coordinates import get_pose_coordinates, save_pose_coordinates
from segmentations    import compute_segmentations, save_segmentations
from gaze_locations   import compute_gaze_locations, save_gaze_locations
from validate         import compute_validation, save_validation

# ============================================================
# CONFIGURE THESE
# ============================================================
HEAD_DILATIONS  = [round(x * 0.1, 1) for x in range(1, 20)]  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HAND_DILATION   = 2.0
ARM_DILATION    = 0.45
ACTIVITY_DILATION = 1.0

MIN_HEAD_RADIUS = 20
MAX_HEAD_RADIUS = 200
MIN_HAND_RADIUS = 20
MAX_HAND_RADIUS = 150
MIN_ARM_THICKNESS = 15
MAX_ARM_THICKNESS = 100

# Root of the project (two levels up from this script)
ROOT = Path(__file__).parent.parent.parent

VIDEO_PATH    = ROOT / "data/input/videos/AM_A1_S5_B2_GA_D1_F1.avi"
GAZE_FOLDER   = ROOT / "data/input/gaze"
OUTPUT_DIR    = ROOT / "data/output/dilation_test"
LANDMARKS_DIR = ROOT / "data/landmarks/frames"
MODEL_PATH    = ROOT / "models/pose_landmarker_heavy.task"
NUM_POSES   = 2
# ============================================================

def extract_id(fname):
    match = re.search(r'~~([^~]+)~~', fname)
    return match.group(1) if match else fname

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    gaze_files = list(GAZE_FOLDER.glob("*.csv"))
    if not gaze_files:
        print("No gaze files found.")
        return

    cap   = cv2.VideoCapture(str(VIDEO_PATH))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # --- Script 1: Run once ---
    pose_csv = LANDMARKS_DIR / f"{VIDEO_PATH.stem}_pose.csv"
    mask_npy = LANDMARKS_DIR / f"{VIDEO_PATH.stem}_masks.npy"

    if pose_csv.exists() and mask_npy.exists():
        print("Pose coordinates already exist, skipping Script 1.")
    else:
        print("Running Script 1: pose coordinates...")
        rows, masks, fps = get_pose_coordinates(VIDEO_PATH, MODEL_PATH, NUM_POSES)
        save_pose_coordinates(rows, masks, pose_csv, mask_npy)

    # Collect results across all dilations for final summary
    # Structure: {head_dilation: {pid: {region: match_pct}}}
    all_results = {}

    for head_dilation in HEAD_DILATIONS:
        dil_str = str(head_dilation).replace(".", "p")
        print(f"\n{'='*50}")
        print(f"Head dilation: {head_dilation}")
        print(f"{'='*50}")

        # --- Script 2: Segmentations (no video) ---
        seg_csv = LANDMARKS_DIR / f"{VIDEO_PATH.stem}_segmentations_hd{dil_str}.csv"
        seg_rows = compute_segmentations(
            pose_csv, vid_w, vid_h,
            head_dilation=head_dilation,
            hand_dilation=HAND_DILATION,
            arm_dilation=ARM_DILATION,
            activity_dilation=ACTIVITY_DILATION,
            min_head_radius=MIN_HEAD_RADIUS,
            max_head_radius=MAX_HEAD_RADIUS,
            min_hand_radius=MIN_HAND_RADIUS,
            max_hand_radius=MAX_HAND_RADIUS,
            min_arm_thickness=MIN_ARM_THICKNESS,
            max_arm_thickness=MAX_ARM_THICKNESS,
        )
        save_segmentations(seg_rows, seg_csv)

        all_results[head_dilation] = {}

        for gaze_path in gaze_files:
            pid = extract_id(gaze_path.name)

            # --- Script 3: Gaze locations ---
            gaze_loc_csv = OUTPUT_DIR / f"{VIDEO_PATH.stem}_{pid}_hd{dil_str}_gaze_locations.csv"
            results = compute_gaze_locations(gaze_path, seg_csv, mask_npy, vid_w, vid_h)
            save_gaze_locations(results, gaze_loc_csv)

            # --- Script 4: Validation ---
            summary, region_stats = compute_validation(gaze_loc_csv, gaze_path)
            if summary:
                val_csv = OUTPUT_DIR / f"{VIDEO_PATH.stem}_{pid}_hd{dil_str}_validation.csv"
                save_validation(summary, region_stats, val_csv)

                all_results[head_dilation][pid] = {
                    "overall": summary["overall_pct"],
                    **{r["region"]: r["match_pct"] for r in region_stats}
                }

    # Collect all region names seen across all dilations and participants
    all_regions = set()
    for dil_data in all_results.values():
        for pid_data in dil_data.values():
            all_regions.update(k for k in pid_data.keys() if k != "overall")

    # --- Final summary ---
    all_regions = sorted(all_regions)
    fieldnames = ["head_dilation", "overall_pct"] + all_regions
    for gaze_path in gaze_files:
        pid = extract_id(gaze_path.name)
        summary_path = OUTPUT_DIR / f"{VIDEO_PATH.stem}_{pid}_dilation_summary.csv"

        rows = []
        for head_dilation in HEAD_DILATIONS:
            pid_data = all_results.get(head_dilation, {}).get(pid, {})
            row = {
                "head_dilation": head_dilation,
                "overall_pct":   pid_data.get("overall", ""),
            }
            for region in all_regions:
                row[region] = pid_data.get(region, "")
            rows.append(row)

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved dilation summary to: {summary_path}")

if __name__ == "__main__":
    main()