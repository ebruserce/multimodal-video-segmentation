import yaml
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

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
MODEL_PATH = ROOT / cfg["paths"]["model"]["pose_landmarker"]
VIDEO_FOLDER = ROOT / cfg["paths"]["data"]["input_videos"]
OUTPUT_FOLDER = ROOT / cfg["paths"]["data"]["output"]

OUTPUT_FOLDER.mkdir(exist_ok=True)

video_files = list(VIDEO_FOLDER.glob(cfg["settings"]["video_input_extension"]))

# For testing purposes, only process the first video file
if video_files:
    video_path = video_files[0]
    print(f"Processing video: {video_path}")
else:
    print("No video files found in the specified folder.")
    exit(1)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    num_poses=2,
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.VIDEO)

# Step 1: Initialize PoseLandmarker object
detector = PoseLandmarker.create_from_options(options)

# Step 2: Load input video
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video")

height, width = frame.shape[:2]

output_path = OUTPUT_FOLDER / f"{video_path.stem}_segmented.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

print(f"Writing output to: {output_path}")
print("Processing frames...")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_index = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame

alpha = 0.7  # transparency level

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int((frame_index / fps) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    overlay = frame.copy()

    if result.segmentation_masks and result.pose_landmarks:

        for pose_landmarks, mp_mask in zip(result.pose_landmarks,
                                        result.segmentation_masks):

            # Function to convert pose landmarks to pixel coordinates
            def to_pixel(idx):
                lm = pose_landmarks[idx]
                return int(lm.x * width), int(lm.y * height)
            
            mask = np.squeeze(mp_mask.numpy_view())
            binary_mask = mask > 0.5

            # ======================
            # HEAD REGION
            # ======================
            head_indices = list(range(0, 11))

            head_points = []
            for idx in head_indices:
                lm = pose_landmarks[idx]
                x = int(lm.x * width)
                y = int(lm.y * height)
                head_points.append([x, y])

            head_points = np.array(head_points, dtype=np.int32)

            # Compute center of head landmarks
            center_x = int(np.mean(head_points[:, 0]))
            center_y = int(np.mean(head_points[:, 1]))

            # Estimate head size from shoulder distance (more stable than ears)
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]

            shoulder_width = abs((left_shoulder.x - right_shoulder.x) * width)

            radius = int(shoulder_width * 0.75)  # adjust scaling factor if needed
            radius = max(radius, 20)  # minimum radius safeguard

            head_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(head_mask, (center_x, center_y), radius, 1, -1)

            # Intersect with segmentation mask
            head_mask = np.logical_and(head_mask == 1, binary_mask)

            # ======================
            # HAND REGION
            # ======================
            lh_indices = [17, 19, 21]
            rh_indices = [18, 20, 22]

            lh_points = []
            for idx in lh_indices:
                lm = pose_landmarks[idx]
                x = int(lm.x * width)
                y = int(lm.y * height)
                lh_points.append([x, y])

            lh_points = np.array(lh_points, dtype=np.int32)
            
            rh_points = []
            for idx in rh_indices:
                lm = pose_landmarks[idx]
                x = int(lm.x * width)
                y = int(lm.y * height)
                rh_points.append([x, y])

            rh_points = np.array(rh_points, dtype=np.int32)

            lh_mask = np.zeros((height, width), dtype=np.uint8)
            rh_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Thickness proportional to distance between thumb and pinky

            lh_width = np.linalg.norm(lh_points[0] - lh_points[-1])
            rh_width = np.linalg.norm(rh_points[0] - rh_points[-1])
            lh_thickness = max(lh_width, 15)
            rh_thickness = max(rh_width, 15)

            # Compute center of hand landmarks
            center_lh_x = int(np.mean(lh_points[:, 0]))
            center_lh_y = int(np.mean(lh_points[:, 1]))
            center_rh_x = int(np.mean(rh_points[:, 0]))
            center_rh_y = int(np.mean(rh_points[:, 1]))

            radius_lh = max(int(lh_width * 2), 20)
            radius_rh = max(int(rh_width * 2), 20)

            cv2.circle(lh_mask, (center_lh_x, center_lh_y), radius_lh, 1, -1)
            cv2.circle(rh_mask, (center_rh_x, center_rh_y), radius_rh, 1, -1)

            # Intersect with segmentation mask
            lh_mask = np.logical_and(lh_mask == 1, binary_mask)
            rh_mask = np.logical_and(rh_mask == 1, binary_mask)

            # ======================
            # ARM REGION
            # ======================          
            arm_mask = np.zeros((height, width), dtype=np.uint8)

            # Thickness proportional to body size
            shoulder_left = to_pixel(11)
            shoulder_right = to_pixel(12)

            body_width = int(abs(shoulder_left[0] - shoulder_right[0]) * 0.45)
            thickness = max(body_width, 15)  # minimum thickness

            # LEFT ARM
            left_points = [11, 13, 15]
            for i in range(len(left_points) - 1):
                p1 = to_pixel(left_points[i])
                p2 = to_pixel(left_points[i+1])
                cv2.line(arm_mask, p1, p2, 1, thickness)

            # RIGHT ARM
            right_points = [12, 14, 16]
            for i in range(len(right_points) - 1):
                p1 = to_pixel(right_points[i])
                p2 = to_pixel(right_points[i+1])
                cv2.line(arm_mask, p1, p2, 1, thickness)

            # Intersect with segmentation mask
            arm_mask = np.logical_and(arm_mask == 1, binary_mask)

            # ======================
            # PRIORITY ENFORCEMENT (HEAD > HANDS > ARMS > BODY)
            # ======================
            lh_mask = np.logical_and(lh_mask, np.logical_not(head_mask))
            rh_mask = np.logical_and(rh_mask, np.logical_not(head_mask))

            arm_mask = np.logical_and(
                arm_mask, 
                np.logical_not(
                    np.logical_or(
                        head_mask, 
                        np.logical_or(
                            lh_mask, rh_mask
                        )
                    )
                )
            )

            body_mask = np.logical_and(
                binary_mask,
                np.logical_not(
                    np.logical_or(
                        head_mask, 
                        np.logical_or(
                            arm_mask, 
                            np.logical_or(
                                lh_mask, rh_mask
                            )
                        )
                    )
                )
            )

            # ======================
            # COLOR LAYERS
            # ======================
            red_layer = np.zeros_like(frame, dtype=np.uint8)
            green_layer = np.zeros_like(frame, dtype=np.uint8)
            blue_layer = np.zeros_like(frame, dtype=np.uint8)
            dark_green_layer = np.zeros_like(frame, dtype=np.uint8)

            red_layer[head_mask] = [0, 0, 255]      # Red
            green_layer[arm_mask] = [0, 255, 0]     # Green
            dark_green_layer[lh_mask] = [0, 128, 0] # Dark Green
            dark_green_layer[rh_mask] = [0, 128, 0] # Dark Green
            blue_layer[body_mask] = [255, 0, 0]     # Blue

            overlay = cv2.addWeighted(overlay, 1.0, blue_layer, alpha, 0)
            overlay = cv2.addWeighted(overlay, 1.0, green_layer, alpha, 0)
            overlay = cv2.addWeighted(overlay, 1.0, dark_green_layer, alpha, 0)
            overlay = cv2.addWeighted(overlay, 1.0, red_layer, alpha, 0)

    out.write(overlay)
    frame_index += 1

    if frame_index % 100 == 0:
        percent = (frame_index / total_frames) * 100
        print(f"{percent:.1f}% complete")

cap.release()
out.release()

print("Processing complete.")
print(f"Saved video to: {output_path}")