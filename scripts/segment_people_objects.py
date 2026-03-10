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

# Settings
lm = {name:i for i, name in enumerate(cfg["pose_landmarks"])}
alpha = cfg["settings"]["alpha"]

# Set up paths
ROOT = Path(cfg["paths"].get("project_root", "."))
POSE_MODEL_PATH = ROOT / cfg["paths"]["models"]["pose_landmarker"]
OBJ_MODEL_PATH = ROOT / cfg["paths"]["models"]["object_detector"]
VIDEO_FOLDER = ROOT / cfg["paths"]["data"]["input_videos"]
OUTPUT_FOLDER = ROOT / cfg["paths"]["data"]["output"]

OUTPUT_FOLDER.mkdir(exist_ok=True)

video_files = list(VIDEO_FOLDER.glob(cfg["settings"]["video_input_extension"]))

# Helper to convert pose landmarks to pixel coordinates
def to_pixel(pose_landmarks, idx):
    landmark = pose_landmarks[idx]
    return int(landmark.x * width), int(landmark.y * height)

# Helper to generate rough ellipse mask for objects
def get_object_mask(frame, bbox):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    center = (bbox.origin_x + bbox.width // 2, bbox.origin_y + bbox.height // 2)
    axes = (bbox.width // 2, bbox.height // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask

# For testing purposes, only process the first video file
if video_files:
    video_path = video_files[0]
    print(f"Processing video: {video_path}")
else:
    print("No video files found in the specified folder.")
    exit(1)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
    num_poses=cfg["settings"]["num_poses"],
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.VIDEO)

obj_options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=str(OBJ_MODEL_PATH)),
    max_results=cfg["settings"]["max_results"],
    score_threshold=cfg["settings"]["score_threshold"],
    category_denylist=["person"],  # To avoid overlap with pose segmentation mask
    running_mode=VisionRunningMode.VIDEO)

# Step 1: Initialize PoseLandmarker and ObjectDetector objects
pose_detector = PoseLandmarker.create_from_options(pose_options)
obj_detector = ObjectDetector.create_from_options(obj_options)

# Step 2: Load input video
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video")

height, width = frame.shape[:2]

output_path = OUTPUT_FOLDER / f"{video_path.stem}_segmented.mp4"
fourcc = cv2.VideoWriter_fourcc(*cfg["settings"]["video_output_extension"])
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

print(f"Writing output to: {output_path}")
print("Processing frames...")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_index = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame

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
    pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)

    overlay = frame.copy()
    all_head_masks = []

    if pose_result.segmentation_masks and pose_result.pose_landmarks:
        for pose_landmarks, mp_mask in zip(pose_result.pose_landmarks,
                                        pose_result.segmentation_masks):
            mask = np.squeeze(mp_mask.numpy_view())
            binary_mask = mask > cfg["settings"]["segmentation_threshold"]

            # Use distance between knees to estimate various thicknesses
            left_knee = pose_landmarks[lm["LEFT_KNEE"]]
            right_knee = pose_landmarks[lm["RIGHT_KNEE"]]
            knee_width = abs((left_knee.x - right_knee.x) * width)

            # ======================
            # HEAD REGION
            # ======================

            head_indices = list(range(lm["NOSE"], lm["LEFT_SHOULDER"]))
            shoulder_indices = [lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"]]

            head_points = []
            for idx in head_indices:
                landmark = pose_landmarks[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                head_points.append([x, y])

            head_points = np.array(head_points, dtype=np.int32)

            shoulder_points = []
            for idx in shoulder_indices:
                landmark = pose_landmarks[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                shoulder_points.append([x,y])
            shoulder_points = np.array(shoulder_points, dtype=np.int32)

            # Compute center of radius using head points for vertical and shoulder for horizontal
            center_x = int(np.mean(shoulder_points[:, 0]))
            center_y = int(np.mean(head_points[:, 1]))

            radius = int(knee_width * 0.6)  # adjust scaling factor if needed
            radius = max(radius, cfg["settings"]["min_radius"])  # minimum radius safeguard

            head_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(head_mask, (center_x, center_y), radius, 1, -1)
            head_mask = head_mask.astype(bool)

            all_head_masks.append(head_mask)

            # ======================
            # HAND REGION
            # ======================
            lh_indices = [lm["LEFT_PINKY"], lm["LEFT_INDEX"], lm["LEFT_THUMB"]]
            rh_indices = [lm["RIGHT_PINKY"], lm["RIGHT_INDEX"], lm["RIGHT_THUMB"]]

            lh_points = []
            for idx in lh_indices:
                landmark = pose_landmarks[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                lh_points.append([x, y])

            lh_points = np.array(lh_points, dtype=np.int32)
            
            rh_points = []
            for idx in rh_indices:
                landmark = pose_landmarks[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                rh_points.append([x, y])

            rh_points = np.array(rh_points, dtype=np.int32)

            lh_mask = np.zeros((height, width), dtype=np.uint8)
            rh_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Thickness proportional to distance between thumb and pinky

            lh_width = np.linalg.norm(lh_points[0] - lh_points[-1])
            rh_width = np.linalg.norm(rh_points[0] - rh_points[-1])

            # Compute center of hand landmarks
            center_lh_x = int(np.mean(lh_points[:, 0]))
            center_lh_y = int(np.mean(lh_points[:, 1]))
            center_rh_x = int(np.mean(rh_points[:, 0]))
            center_rh_y = int(np.mean(rh_points[:, 1]))

            radius_lh = max(int(lh_width * 2), cfg["settings"]["min_radius"])
            radius_rh = max(int(rh_width * 2), cfg["settings"]["min_radius"])

            cv2.circle(lh_mask, (center_lh_x, center_lh_y), radius_lh, 1, -1)
            cv2.circle(rh_mask, (center_rh_x, center_rh_y), radius_rh, 1, -1)

            lh_mask = lh_mask.astype(bool)
            rh_mask = rh_mask.astype(bool)

            # ======================
            # ARM REGION
            # ======================          
            arm_mask = np.zeros((height, width), dtype=np.uint8)
            thickness = max(int(knee_width * 0.45), 15)

            # LEFT ARM
            left_points = [lm["LEFT_SHOULDER"], lm["LEFT_ELBOW"], lm["LEFT_WRIST"]]
            for i in range(len(left_points) - 1):
                p1 = to_pixel(pose_landmarks, left_points[i])
                p2 = to_pixel(pose_landmarks, left_points[i+1])
                cv2.line(arm_mask, p1, p2, 1, thickness)

            # RIGHT ARM
            right_points = [lm["RIGHT_SHOULDER"], lm["RIGHT_ELBOW"], lm["RIGHT_WRIST"]]
            for i in range(len(right_points) - 1):
                p1 = to_pixel(pose_landmarks, right_points[i])
                p2 = to_pixel(pose_landmarks, right_points[i+1])
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

            red_layer[head_mask.astype(bool)] = [0, 0, 255]      # Red
            green_layer[arm_mask.astype(bool)] = [0, 255, 0]     # Green
            dark_green_layer[lh_mask.astype(bool)] = [0, 128, 0] # Dark Green
            dark_green_layer[rh_mask.astype(bool)] = [0, 128, 0] # Dark Green
            blue_layer[body_mask.astype(bool)] = [255, 0, 0]     # Blue

            overlay = cv2.addWeighted(overlay, 1.0, blue_layer, alpha, 0)
            overlay = cv2.addWeighted(overlay, 1.0, green_layer, alpha, 0)
            overlay = cv2.addWeighted(overlay, 1.0, dark_green_layer, alpha, 0)
            overlay = cv2.addWeighted(overlay, 1.0, red_layer, alpha, 0)

    obj_result = obj_detector.detect_for_video(mp_image, timestamp_ms)
    # Build combined object mask from all detections
    obj_combined_mask = np.zeros((height, width), dtype=np.uint8)
    for detection in obj_result.detections:
        obj_mask = get_object_mask(frame, detection.bounding_box)
        obj_combined_mask = np.logical_or(obj_combined_mask, obj_mask)

    # Remove person regions from object mask
    if pose_result.segmentation_masks:
        person_combined = np.zeros((height, width), dtype=bool)
        for mp_mask in pose_result.segmentation_masks:
            mask = np.squeeze(mp_mask.numpy_view())
            person_combined = np.logical_or(person_combined, mask > cfg["settings"]["segmentation_threshold"])
        for hm in all_head_masks:
                person_combined = np.logical_or(person_combined, hm)
        obj_combined_mask = np.logical_and(obj_combined_mask, np.logical_not(person_combined))

    # Draw object segmentations onto overlay
    yellow_layer = np.zeros_like(frame, dtype=np.uint8)
    yellow_layer[obj_combined_mask] = [0, 255, 255]  # BGR yellow
    overlay = cv2.addWeighted(overlay, 1.0, yellow_layer, alpha, 0)

    out.write(overlay)
    frame_index += 1

    if frame_index % 100 == 0:
        percent = (frame_index / total_frames) * 100
        print(f"{percent:.1f}% complete")

cap.release()
out.release()

print("Processing complete.")
print(f"Saved video to: {output_path}")