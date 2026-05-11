import csv
import yaml
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


def load_config():
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    local_cfg_path = Path("configs/paths.local.yaml")
    if local_cfg_path.exists():
        with open(local_cfg_path) as f:
            local = yaml.safe_load(f)
        cfg["paths"].update(local["paths"])
    return cfg


def get_pose_coordinates(video_path, model_path, num_poses=2):
    """
    Runs MediaPipe pose detection on every frame of a video.
    Returns:
      - rows: list of dicts with raw video-space landmark coordinates per frame
      - masks: np.ndarray of shape (total_frames, vid_h, vid_w) — binary segmentation masks
      - fps: float
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_poses=num_poses,
        output_segmentation_masks=True,
        running_mode=VisionRunningMode.VIDEO)

    detector = PoseLandmarker.create_from_options(options)

    rows = []
    # Pre-allocate mask array — bool dtype keeps memory reasonable
    masks = np.zeros((total_frames, vid_h, vid_w), dtype=bool)
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_index / fps) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        row = {
            "frame":        frame_index,
            "timestamp_ms": timestamp_ms,
            "vid_w":        vid_w,
            "vid_h":        vid_h,
            "detected":     0,
        }

        # Combine segmentation masks from all detected people
        if result.segmentation_masks:
            combined = np.zeros((vid_h, vid_w), dtype=bool)
            for mp_mask in result.segmentation_masks:
                mask = np.squeeze(mp_mask.numpy_view()) > 0.5
                combined = np.logical_or(combined, mask)
            masks[frame_index] = combined

        if result.pose_landmarks and len(result.pose_landmarks) == 2:
            sorted_poses = sorted(
                result.pose_landmarks,
                key=lambda p: (p[0].x + p[1].x) / 2
            )
            for person_idx, pose_landmarks in enumerate(sorted_poses):
                label = "a1" if person_idx == 0 else "a2"
                for lm_idx, landmark in enumerate(pose_landmarks):
                    row[f"{label}_lm{lm_idx}_x"]   = round(landmark.x * vid_w, 4)
                    row[f"{label}_lm{lm_idx}_y"]   = round(landmark.y * vid_h, 4)
                    row[f"{label}_lm{lm_idx}_vis"] = round(landmark.visibility, 4)
            row["detected"] = 1

        rows.append(row)
        frame_index += 1

        if frame_index % 100 == 0:
            pct = (frame_index / total_frames) * 100
            print(f"  {pct:.1f}% complete")

    cap.release()
    detector.close()

    # Trim mask array in case video ended early
    masks = masks[:frame_index]

    return rows, masks, fps


def save_pose_coordinates(rows, masks, output_csv_path, output_mask_path):
    """Save pose coordinate rows to CSV and masks to .npy."""
    if not rows:
        print("No rows to save.")
        return

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved pose coordinates to: {output_csv_path}")

    np.save(str(output_mask_path), masks)
    print(f"Saved segmentation masks to: {output_mask_path} "
          f"(shape: {masks.shape}, dtype: {masks.dtype})")


if __name__ == "__main__":
    cfg = load_config()
    ROOT             = Path(cfg["paths"].get("project_root", "."))
    model_path       = ROOT / cfg["paths"]["models"]["pose_landmarker"]
    video_folder     = ROOT / cfg["paths"]["data"]["input_videos"]
    landmarks_folder = ROOT / cfg["paths"]["data"]["landmarks"]
    frames_folder    = landmarks_folder / "frames"
    frames_folder.mkdir(parents=True, exist_ok=True)

    video_files = list(video_folder.glob(cfg["settings"]["video_input_extension"]))
    if not video_files:
        print("No video files found.")
        exit(1)

    video_path = video_files[0]
    print(f"Processing: {video_path.stem}")

    rows, masks, fps = get_pose_coordinates(
        video_path, model_path, cfg["settings"]["num_poses"]
    )
    save_pose_coordinates(
        rows, masks,
        frames_folder / f"{video_path.stem}_pose.csv",
        frames_folder / f"{video_path.stem}_masks.npy",
    )
    print("Done.")