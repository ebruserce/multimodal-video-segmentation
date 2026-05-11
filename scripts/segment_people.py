import yaml
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
import re

# Load configs
with open("configs/base.yaml") as f:
    cfg = yaml.safe_load(f)

local_cfg_path = Path("configs/paths.local.yaml")
if local_cfg_path.exists():
    with open(local_cfg_path) as f:
        local = yaml.safe_load(f)
    cfg["paths"].update(local["paths"])

# Set up paths
ROOT = Path(cfg["paths"].get("project_root", "."))
MODEL_PATH = ROOT / cfg["paths"]["models"]["pose_landmarker"]
VIDEO_FOLDER = ROOT / cfg["paths"]["data"]["input_videos"]
OUTPUT_FOLDER = ROOT / cfg["paths"]["data"]["output"]
GAZE_FOLDER = ROOT / cfg["paths"]["data"]["gaze_folder"]

OUTPUT_FOLDER.mkdir(exist_ok=True)

CANVAS_W, CANVAS_H = 1680, 1050
BG_COLOR = (211, 211, 211)  # light gray, matches Brenda's default

def get_canvas_offsets(vid_w, vid_h, canvas_w, canvas_h):
    """Compute canvas and video crop coordinates for centering, matching Brenda's approach."""
    x_offset = (canvas_w - vid_w) // 2
    y_offset = (canvas_h - vid_h) // 2

    cy1 = max(0, y_offset)
    cy2 = min(canvas_h, y_offset + vid_h)
    cx1 = max(0, x_offset)
    cx2 = min(canvas_w, x_offset + vid_w)

    vy1 = max(0, -y_offset)
    vy2 = vy1 + (cy2 - cy1)
    vx1 = max(0, -x_offset)
    vx2 = vx1 + (cx2 - cx1)

    return cx1, cx2, cy1, cy2, vx1, vx2, vy1, vy2

def extract_participant_id(filename):
    """Extract participant ID from between the first pair of double tildes."""
    match = re.search(r'~~([^~]+)~~', filename)
    return match.group(1) if match else filename

def load_gaze(csv_path):
    """Load and filter a gaze CSV, return lookup dict of ms -> (sx, sy)."""
    df = pd.read_csv(csv_path)
    df = df[df["valid"] == 1].copy()
    df = df.groupby("t", as_index=False).first()
    return {
        int(row["t"]): (float(row["sx"]), float(row["sy"]))
        for _, row in df.iterrows()
    }

video_files = list(VIDEO_FOLDER.glob(cfg["settings"]["video_input_extension"]))
gaze_files  = list(GAZE_FOLDER.glob("*.csv"))

if not video_files:
    print("No video files found.")
    exit(1)

if not gaze_files:
    print("No gaze CSV files found.")
    exit(1)

video_path = video_files[0]
print(f"Video: {video_path}")
print(f"Found {len(gaze_files)} gaze file(s).")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

for gaze_path in gaze_files:
    participant_id = extract_participant_id(gaze_path.name)
    print(f"\nProcessing gaze file: {gaze_path.name}")
    print(f"Participant ID: {participant_id}")

    gaze_lookup = load_gaze(gaze_path)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_poses=2,
        output_segmentation_masks=True,
        running_mode=VisionRunningMode.VIDEO)

    detector = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute centering offsets once per video
    cx1, cx2, cy1, cy2, vx1, vx2, vy1, vy2 = get_canvas_offsets(vid_w, vid_h, CANVAS_W, CANVAS_H)

    ret, _ = cap.read()
    if not ret:
        print(f"Could not read video, skipping.")
        cap.release()
        detector.close()
        continue

    output_path = OUTPUT_FOLDER / f"{video_path.stem}_{participant_id}_gaze.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (CANVAS_W, CANVAS_H))

    print(f"Writing output to: {output_path}")
    print("Processing frames...")

    frame_index = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    alpha = cfg["settings"]["alpha"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Build canvas and center video frame in it, matching Brenda's approach
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
        canvas[cy1:cy2, cx1:cx2] = frame[vy1:vy2, vx1:vx2]

        # Run pose detection on the original frame (not canvas)
        # so landmark coordinates are in video space, then offset to canvas space
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int((frame_index / fps) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        overlay = canvas.copy()

        if result.segmentation_masks and result.pose_landmarks:
            for pose_landmarks, mp_mask in zip(result.pose_landmarks,
                                               result.segmentation_masks):

                def to_pixel(idx):
                    """Convert landmark to pixel coords in canvas space."""
                    lm = pose_landmarks[idx]
                    x = int(lm.x * vid_w) + cx1
                    y = int(lm.y * vid_h) + cy1
                    return x, y

                # Segmentation mask is in video space, place it onto canvas
                mask = np.squeeze(mp_mask.numpy_view())
                binary_mask_vid = mask > 0.5

                binary_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
                binary_mask[cy1:cy2, cx1:cx2] = binary_mask_vid[vy1:vy2, vx1:vx2]

                # HEAD REGION
                head_indices = list(range(0, 11))
                head_points = []
                for idx in head_indices:
                    lm = pose_landmarks[idx]
                    x = int(lm.x * vid_w) + cx1
                    y = int(lm.y * vid_h) + cy1
                    head_points.append([x, y])

                head_points = np.array(head_points, dtype=np.int32)
                center_x = int(np.mean(head_points[:, 0]))
                center_y = int(np.mean(head_points[:, 1]))

                left_shoulder  = pose_landmarks[11]
                right_shoulder = pose_landmarks[12]
                shoulder_width = abs((left_shoulder.x - right_shoulder.x) * vid_w)

                radius = int(shoulder_width * 0.75)
                radius = max(radius, 20)

                head_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
                cv2.circle(head_mask, (center_x, center_y), radius, 1, -1)
                head_mask = np.logical_and(head_mask == 1, binary_mask)

                # HAND REGION
                lh_indices = [17, 19, 21]
                rh_indices = [18, 20, 22]

                lh_points = np.array([[int(pose_landmarks[i].x * vid_w) + cx1,
                                       int(pose_landmarks[i].y * vid_h) + cy1] for i in lh_indices], dtype=np.int32)
                rh_points = np.array([[int(pose_landmarks[i].x * vid_w) + cx1,
                                       int(pose_landmarks[i].y * vid_h) + cy1] for i in rh_indices], dtype=np.int32)

                lh_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
                rh_mask = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)

                lh_width = np.linalg.norm(lh_points[0] - lh_points[-1])
                rh_width = np.linalg.norm(rh_points[0] - rh_points[-1])

                center_lh_x = int(np.mean(lh_points[:, 0]))
                center_lh_y = int(np.mean(lh_points[:, 1]))
                center_rh_x = int(np.mean(rh_points[:, 0]))
                center_rh_y = int(np.mean(rh_points[:, 1]))

                radius_lh = max(int(lh_width * 2), 20)
                radius_rh = max(int(rh_width * 2), 20)

                cv2.circle(lh_mask, (center_lh_x, center_lh_y), radius_lh, 1, -1)
                cv2.circle(rh_mask, (center_rh_x, center_rh_y), radius_rh, 1, -1)

                lh_mask = np.logical_and(lh_mask == 1, binary_mask)
                rh_mask = np.logical_and(rh_mask == 1, binary_mask)

                # ARM REGION
                arm_mask  = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
                shoulder_left  = to_pixel(11)
                shoulder_right = to_pixel(12)
                body_width = int(abs(shoulder_left[0] - shoulder_right[0]) * 0.45)
                thickness  = max(body_width, 15)

                for i in range(len([11, 13, 15]) - 1):
                    cv2.line(arm_mask, to_pixel([11, 13, 15][i]), to_pixel([11, 13, 15][i+1]), 1, thickness)
                for i in range(len([12, 14, 16]) - 1):
                    cv2.line(arm_mask, to_pixel([12, 14, 16][i]), to_pixel([12, 14, 16][i+1]), 1, thickness)

                arm_mask = np.logical_and(arm_mask == 1, binary_mask)

                # PRIORITY ENFORCEMENT
                lh_mask = np.logical_and(lh_mask, np.logical_not(head_mask))
                rh_mask = np.logical_and(rh_mask, np.logical_not(head_mask))
                arm_mask = np.logical_and(
                    arm_mask,
                    np.logical_not(np.logical_or(head_mask, np.logical_or(lh_mask, rh_mask)))
                )
                body_mask = np.logical_and(
                    binary_mask,
                    np.logical_not(
                        np.logical_or(head_mask,
                            np.logical_or(arm_mask, np.logical_or(lh_mask, rh_mask)))
                    )
                )

                # COLOR LAYERS
                red_layer        = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                green_layer      = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                blue_layer       = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                dark_green_layer = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

                red_layer[head_mask]      = [0, 0, 255]
                green_layer[arm_mask]     = [0, 255, 0]
                dark_green_layer[lh_mask] = [0, 128, 0]
                dark_green_layer[rh_mask] = [0, 128, 0]
                blue_layer[body_mask]     = [255, 0, 0]

                overlay = cv2.addWeighted(overlay, 1.0, blue_layer,       alpha, 0)
                overlay = cv2.addWeighted(overlay, 1.0, green_layer,      alpha, 0)
                overlay = cv2.addWeighted(overlay, 1.0, dark_green_layer, alpha, 0)
                overlay = cv2.addWeighted(overlay, 1.0, red_layer,        alpha, 0)

        # Draw gaze point for this frame
        if gaze_lookup:
            closest_t = min(gaze_lookup.keys(), key=lambda t: abs(t - timestamp_ms))
            if abs(closest_t - timestamp_ms) <= (1000 / fps) / 2:
                gx, gy = gaze_lookup[closest_t]
                gx, gy = int(gx), int(gy)
                cv2.circle(overlay, (gx, gy), 15, (255, 0, 255), 2)
                cv2.circle(overlay, (gx, gy), 3,  (255, 0, 255), -1)

        out.write(overlay)
        frame_index += 1

        if frame_index % 100 == 0:
            percent = (frame_index / total_frames) * 100
            print(f"  {percent:.1f}% complete")

    cap.release()
    out.release()
    detector.close()
    print(f"Done: {output_path}")

print("\nAll gaze files processed.")