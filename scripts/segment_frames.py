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

# Load configs
with open("configs/base.yaml") as f:
    cfg = yaml.safe_load(f)

local_cfg_path = Path("configs/paths.local.yaml")
if local_cfg_path.exists():
    with open(local_cfg_path) as f:
        local = yaml.safe_load(f)
    cfg["paths"].update(local["paths"])

# Settings
lm = {name: i for i, name in enumerate(cfg["pose_landmarks"])}
alpha = cfg["settings"]["alpha"]

# Set up paths
ROOT = Path(cfg["paths"].get("project_root", "."))
POSE_MODEL_PATH = ROOT / cfg["paths"]["models"]["pose_landmarker"]
VIDEO_FOLDER = ROOT / cfg["paths"]["data"]["input_videos"]
OUTPUT_FOLDER = ROOT / cfg["paths"]["data"]["output"]
LANDMARKS_FOLDER = ROOT / cfg["paths"]["data"]["landmarks"]

OUTPUT_FOLDER.mkdir(exist_ok=True)
LANDMARKS_FOLDER.mkdir(exist_ok=True)

video_files = list(VIDEO_FOLDER.glob(cfg["settings"]["video_input_extension"]))

if not video_files:
    print("No video files found in the specified folder.")
    exit(1)

# Helpers to convert pose landmarks to pixel coordinates
def to_pixel(pose_landmarks, idx, width, height):
    landmark = pose_landmarks[idx]
    return [int(landmark.x * width), int(landmark.y * height)]

def to_positions(pose_landmarks, indices, width, height, d_type=np.uint8):
    points = []
    for idx in indices:
        points.append(to_pixel(pose_landmarks, idx, width, height))
    return np.array(points, dtype=d_type)

def get_center_x(pose_landmarks):
    return (pose_landmarks[lm["LEFT_SHOULDER"]].x + pose_landmarks[lm["RIGHT_SHOULDER"]].x) / 2

def get_head_center(pose_landmarks, width, height):
    head_indices = list(range(lm["NOSE"], lm["LEFT_SHOULDER"]))
    shoulder_indices = [lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"]]

    head_points = to_positions(pose_landmarks, head_indices, width, height, d_type=np.int32)
    shoulder_points = to_positions(pose_landmarks, shoulder_indices, width, height, d_type=np.int32)

    center_x = int(np.mean(shoulder_points[:, 0]))
    center_y = int(np.mean(head_points[:, 1]))
    return (center_x, center_y)

def get_activity_mask(left_pose, right_pose, height, width):
    x_left  = int(left_pose[lm["LEFT_ELBOW"]].x * width)
    x_right = int(right_pose[lm["RIGHT_ELBOW"]].x * width)

    y_top = int(min(
        (left_pose[lm["LEFT_SHOULDER"]].y + left_pose[lm["LEFT_HIP"]].y) / 2 * height,
        (right_pose[lm["RIGHT_SHOULDER"]].y + right_pose[lm["RIGHT_HIP"]].y) / 2 * height
    ))
    y_bottom = int(max(
        left_pose[lm["LEFT_FOOT_INDEX"]].y * height,
        right_pose[lm["RIGHT_FOOT_INDEX"]].y * height
    ))

    x_left   = max(0, x_left)
    x_right  = min(width, x_right)
    y_top    = max(0, y_top)
    y_bottom = min(height, y_bottom)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (x_left, y_top), (x_right, y_bottom), 1, -1)
    return mask.astype(bool)

# Initialize PoseLandmarker in IMAGE mode since we're processing single frames
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
    num_poses=cfg["settings"]["num_poses"],
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.IMAGE)

pose_detector = PoseLandmarker.create_from_options(pose_options)

# One row per video, all written at the end
landmarks_rows = []

for video_path in video_files:
    print(f"\nProcessing: {video_path.stem}")

    # Read only the first frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read {video_path.stem}, skipping.")
        continue

    height, width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run pose detection on single image (no timestamp needed in IMAGE mode)
    pose_result = pose_detector.detect(mp_image)

    overlay = np.zeros_like(frame)

    if not pose_result.segmentation_masks or not pose_result.pose_landmarks:
        print(f"No poses detected in {video_path.stem}, skipping.")
        continue

    sorted_poses = sorted(
        zip(pose_result.pose_landmarks, pose_result.segmentation_masks),
        key=lambda p: get_center_x(p[0])
    )

    if len(sorted_poses) != 2:
        print(f"Expected 2 people in {video_path.stem}, found {len(sorted_poses)}, skipping.")
        continue

    (left_pose, left_seg), (right_pose, right_seg) = sorted_poses

    # Compute head centers for this frame
    left_head  = get_head_center(left_pose, width, height)
    right_head = get_head_center(right_pose, width, height)

    for pose_landmarks, mp_mask in [(left_pose, left_seg), (right_pose, right_seg)]:
        mask = np.squeeze(mp_mask.numpy_view())
        binary_mask = mask > cfg["settings"]["segmentation_threshold"]

        kernel_size = cfg["settings"]["mask_dilation_kernel"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        left_knee  = pose_landmarks[lm["LEFT_KNEE"]]
        right_knee = pose_landmarks[lm["RIGHT_KNEE"]]
        knee_width = abs((left_knee.x - right_knee.x) * width)

        head_indices    = list(range(lm["NOSE"], lm["LEFT_SHOULDER"]))
        shoulder_indices = [lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"]]

        head_points     = to_positions(pose_landmarks, head_indices, width, height, d_type=np.int32)
        shoulder_points = to_positions(pose_landmarks, shoulder_indices, width, height, d_type=np.int32)

        center_x = int(np.mean(shoulder_points[:, 0]))
        center_y = int(np.mean(head_points[:, 1]))

        radius = int(knee_width * 0.6)
        radius = max(radius, cfg["settings"]["min_radius"])

        head_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(head_mask, (center_x, center_y), radius, 1, -1)
        head_mask = head_mask.astype(bool)

        lh_indices = [lm["LEFT_PINKY"], lm["LEFT_INDEX"], lm["LEFT_THUMB"]]
        rh_indices = [lm["RIGHT_PINKY"], lm["RIGHT_INDEX"], lm["RIGHT_THUMB"]]

        lh_points = to_positions(pose_landmarks, lh_indices, width, height, d_type=np.int32)
        rh_points = to_positions(pose_landmarks, rh_indices, width, height, d_type=np.int32)

        lh_mask = np.zeros((height, width), dtype=np.uint8)
        rh_mask = np.zeros((height, width), dtype=np.uint8)

        lh_width = np.linalg.norm(lh_points[0] - lh_points[-1])
        rh_width = np.linalg.norm(rh_points[0] - rh_points[-1])

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

        arm_mask  = np.zeros((height, width), dtype=np.uint8)
        thickness = max(int(knee_width * 0.45), 15)

        left_points = [lm["LEFT_SHOULDER"], lm["LEFT_ELBOW"], lm["LEFT_WRIST"]]
        for i in range(len(left_points) - 1):
            p1 = to_pixel(pose_landmarks, left_points[i], width, height)
            p2 = to_pixel(pose_landmarks, left_points[i+1], width, height)
            cv2.line(arm_mask, p1, p2, 1, thickness)

        right_points = [lm["RIGHT_SHOULDER"], lm["RIGHT_ELBOW"], lm["RIGHT_WRIST"]]
        for i in range(len(right_points) - 1):
            p1 = to_pixel(pose_landmarks, right_points[i], width, height)
            p2 = to_pixel(pose_landmarks, right_points[i+1], width, height)
            cv2.line(arm_mask, p1, p2, 1, thickness)

        lh_mask = np.logical_and(lh_mask, np.logical_not(head_mask))
        rh_mask = np.logical_and(rh_mask, np.logical_not(head_mask))

        arm_mask = np.logical_and(
            arm_mask,
            np.logical_not(np.logical_or(head_mask, np.logical_or(lh_mask, rh_mask)))
        )

        body_mask = np.logical_and(
            binary_mask,
            np.logical_not(
                np.logical_or(
                    head_mask,
                    np.logical_or(arm_mask, np.logical_or(lh_mask, rh_mask))
                )
            )
        )

        red_layer        = np.zeros_like(frame, dtype=np.uint8)
        green_layer      = np.zeros_like(frame, dtype=np.uint8)
        blue_layer       = np.zeros_like(frame, dtype=np.uint8)
        dark_green_layer = np.zeros_like(frame, dtype=np.uint8)

        red_layer[head_mask]      = [0, 0, 255]
        green_layer[arm_mask]     = [0, 255, 0]
        dark_green_layer[lh_mask] = [0, 128, 0]
        dark_green_layer[rh_mask] = [0, 128, 0]
        blue_layer[body_mask]     = [255, 0, 0]

        overlay = cv2.addWeighted(overlay, 1.0, blue_layer,       alpha, 0)
        overlay = cv2.addWeighted(overlay, 1.0, green_layer,      alpha, 0)
        overlay = cv2.addWeighted(overlay, 1.0, dark_green_layer, alpha, 0)
        overlay = cv2.addWeighted(overlay, 1.0, red_layer,        alpha, 0)

    activity_mask = get_activity_mask(left_pose, right_pose, height, width)
    yellow_layer  = np.zeros_like(frame, dtype=np.uint8)
    yellow_layer[activity_mask] = [0, 255, 255]
    overlay = cv2.addWeighted(overlay, 1.0, yellow_layer, alpha, 0)

    # Save segmented image
    output_path = OUTPUT_FOLDER / f"{video_path.stem}_segmented.jpg"
    cv2.imwrite(str(output_path), overlay)
    print(f"Saved segmented image to: {output_path}")

    landmarks_rows.append({
        "video":          video_path.stem,
        "left_head_x":    left_head[0],
        "left_head_y":    left_head[1],
        "right_head_x":   right_head[0],
        "right_head_y":   right_head[1],
    })

pose_detector.close()

# Save all landmarks to a single CSV
landmarks_path = LANDMARKS_FOLDER / "head_centers.csv"
with open(landmarks_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "video", "left_head_x", "left_head_y", "right_head_x", "right_head_y"
    ])
    writer.writeheader()
    writer.writerows(landmarks_rows)

print(f"\nSaved head centers to: {landmarks_path}")
print("All done.")