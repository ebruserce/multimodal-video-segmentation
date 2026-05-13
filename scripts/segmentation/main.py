# Ebru Serce
import cv2
import re
import yaml
from pathlib import Path

from scripts.segmentation.pose_coordinates import get_pose_coordinates, save_pose_coordinates
from scripts.segmentation.segmentations   import compute_segmentations, save_segmentations, render_segmentation_video
from scripts.segmentation.gaze_locations  import compute_gaze_locations, save_gaze_locations
from scripts.segmentation.validate        import compute_validation, save_validation


def load_config():
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    local_cfg_path = Path("configs/paths.local.yaml")
    if local_cfg_path.exists():
        with open(local_cfg_path) as f:
            local = yaml.safe_load(f)
        cfg["paths"].update(local["paths"])
    return cfg

def extract_id(fname):
    match = re.search(r'~~([^~]+)~~', fname)
    return match.group(1) if match else fname


def run_pipeline(video_path, gaze_paths, cfg):
    ROOT             = Path(cfg["paths"].get("project_root", "."))
    model_path       = ROOT / cfg["paths"]["models"]["pose_landmarker"]
    landmarks_folder = ROOT / cfg["paths"]["data"]["landmarks"]
    frames_folder    = landmarks_folder / "frames"
    output_folder    = ROOT / cfg["paths"]["data"]["output"]

    frames_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap   = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # --- Script 1: Pose coordinates ---
    print(f"\n[1/4] Extracting pose coordinates: {video_path.stem}")
    pose_csv = frames_folder / f"{video_path.stem}_pose.csv"
    mask_npy = frames_folder / f"{video_path.stem}_masks.npy"
    rows, masks, fps = get_pose_coordinates(
        video_path, model_path, cfg["settings"]["num_poses"]
    )
    save_pose_coordinates(rows, masks, pose_csv, mask_npy)

    # --- Script 2: Segmentations ---
    print(f"\n[2/4] Computing segmentations: {video_path.stem}")
    seg_csv   = frames_folder / f"{video_path.stem}_segmentations.csv"
    seg_video = output_folder / f"{video_path.stem}_segmentation.mp4"

    seg_rows = compute_segmentations(
        pose_csv, vid_w, vid_h,
        head_dilation=cfg["settings"].get("head_dilation", 0.75),
        hand_dilation=cfg["settings"].get("hand_dilation", 2.0),
        arm_dilation=cfg["settings"].get("arm_dilation", 0.45),
        activity_dilation=cfg["settings"].get("activity_dilation", 1.0),
        min_head_radius=cfg["settings"]["min_head_radius"],
        max_head_radius=cfg["settings"]["max_head_radius"],
        min_hand_radius=cfg["settings"]["min_hand_radius"],
        max_hand_radius=cfg["settings"]["max_hand_radius"],
        min_arm_thickness=cfg["settings"]["min_arm_thickness"],
        max_arm_thickness=cfg["settings"]["max_arm_thickness"],
    )
    save_segmentations(seg_rows, seg_csv)
    render_segmentation_video(
        video_path, seg_csv, mask_npy, seg_video,
        video_alpha=cfg["settings"].get("seg_video_alpha", 0.0)
    )

    # --- Scripts 3 & 4: Gaze locations + validation per gaze file ---
    for gaze_path in gaze_paths:
        pid = extract_id(gaze_path.name)
        print(f"\n[3/4] Gaze locations: {pid}")
        gaze_loc_csv = output_folder / f"{video_path.stem}_{pid}_gaze_locations.csv"
        results = compute_gaze_locations(gaze_path, seg_csv, mask_npy, vid_w, vid_h)  # added mask_npy
        save_gaze_locations(results, gaze_loc_csv)

        print(f"[4/4] Validation: {pid}")
        summary, region_stats = compute_validation(gaze_loc_csv, gaze_path)
        if summary:
            val_csv = output_folder / f"{video_path.stem}_{pid}_validation.csv"
            save_validation(summary, region_stats, val_csv)


if __name__ == "__main__":
    cfg = load_config()
    ROOT         = Path(cfg["paths"].get("project_root", "."))
    video_folder = ROOT / cfg["paths"]["data"]["input_videos"]
    gaze_folder  = ROOT / cfg["paths"]["data"]["gaze_folder"]

    video_files = list(video_folder.glob(cfg["settings"]["video_input_extension"]))
    gaze_files  = list(gaze_folder.glob("*.csv"))

    if not video_files:
        print("No video files found.")
        exit(1)
    if not gaze_files:
        print("No gaze files found.")
        exit(1)

    run_pipeline(video_files[0], gaze_files, cfg)
    print("\nAll done.")