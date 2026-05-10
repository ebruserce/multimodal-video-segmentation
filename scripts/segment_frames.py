#!/usr/bin/env python3

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
TARGET_W, TARGET_H = 1680, 1050
DILATION_LEVELS = [round(x * 0.1, 1) for x in range(5, 11)]  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Set up paths
ROOT = Path(cfg["paths"].get("project_root", "."))
POSE_MODEL_PATH = ROOT / cfg["paths"]["models"]["pose_landmarker"]
VIDEO_FOLDER = ROOT / cfg["paths"]["data"]["input_videos"]
OUTPUT_FOLDER = ROOT / cfg["paths"]["data"]["output"]
LANDMARKS_FOLDER = ROOT / cfg["paths"]["data"]["landmarks"]
FRAMES_FOLDER = LANDMARKS_FOLDER / "frames"

OUTPUT_FOLDER.mkdir(exist_ok=True)
LANDMARKS_FOLDER.mkdir(exist_ok=True)
FRAMES_FOLDER.mkdir(exist_ok=True)

video_files = list(VIDEO_FOLDER.glob(cfg["settings"]["video_input_extension"]))

if not video_files:
    print("No video files found in the specified folder.")
    exit(1)

def resize_to_canvas(frame, target_w, target_h):
    """Resize frame to fit within target dimensions, centered on a black canvas."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

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

def get_head_circle(pose_landmarks, width, height, dilation):
    """Returns (center_x, center_y, radius) for a given dilation level."""
    head_indices     = list(range(lm["NOSE"], lm["LEFT_SHOULDER"]))
    shoulder_indices = [lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"]]

    head_points     = to_positions(pose_landmarks, head_indices, width, height, d_type=np.int32)
    shoulder_points = to_positions(pose_landmarks, shoulder_indices, width, height, d_type=np.int32)

    center_x = int(np.mean(shoulder_points[:, 0]))
    center_y = int(np.mean(head_points[:, 1]))

    left_knee  = pose_landmarks[lm["LEFT_KNEE"]]
    right_knee = pose_landmarks[lm["RIGHT_KNEE"]]
    knee_width = abs((left_knee.x - right_knee.x) * width)

    radius = int(knee_width * dilation)
    radius = max(radius, cfg["settings"]["min_head_radius"])
    radius = min(radius, cfg["settings"]["max_head_radius"])

    return (center_x, center_y, radius)

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

def process_first_frame(frame, pose_detector, width, height):
    """Runs segmentation on a single frame and returns the overlay image."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    pose_result = pose_detector.detect(mp_image)

    if not pose_result.segmentation_masks or not pose_result.pose_landmarks:
        return None

    sorted_poses = sorted(
        zip(pose_result.pose_landmarks, pose_result.segmentation_masks),
        key=lambda p: get_center_x(p[0])
    )

    if len(sorted_poses) != 2:
        return None

    (left_pose, left_seg), (right_pose, right_seg) = sorted_poses
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    for pose_landmarks, mp_mask in [(left_pose, left_seg), (right_pose, right_seg)]:
        mask = np.squeeze(mp_mask.numpy_view())
        binary_mask = mask > cfg["settings"]["segmentation_threshold"]

        kernel_size = cfg["settings"]["mask_dilation_kernel"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_mask = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        left_knee  = pose_landmarks[lm["LEFT_KNEE"]]
        right_knee = pose_landmarks[lm["RIGHT_KNEE"]]
        knee_width = abs((left_knee.x - right_knee.x) * width)

        head_indices     = list(range(lm["NOSE"], lm["LEFT_SHOULDER"]))
        shoulder_indices = [lm["LEFT_SHOULDER"], lm["RIGHT_SHOULDER"]]

        head_points     = to_positions(pose_landmarks, head_indices, width, height, d_type=np.int32)
        shoulder_points = to_positions(pose_landmarks, shoulder_indices, width, height, d_type=np.int32)

        center_x = int(np.mean(shoulder_points[:, 0]))
        center_y = int(np.mean(head_points[:, 1]))

        radius = int(knee_width * 0.6)  # default dilation for visual
        radius = max(radius, cfg["settings"]["min_head_radius"])
        radius = min(radius, cfg["settings"]["max_head_radius"])

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

        radius_lh = max(int(lh_width * 2), cfg["settings"]["min_hand_radius"])
        radius_lh = min(radius_lh, cfg["settings"]["max_hand_radius"])
        radius_rh = max(int(rh_width * 2), cfg["settings"]["min_hand_radius"])
        radius_rh = min(radius_rh, cfg["settings"]["max_hand_radius"])

        cv2.circle(lh_mask, (center_lh_x, center_lh_y), radius_lh, 1, -1)
        cv2.circle(rh_mask, (center_rh_x, center_rh_y), radius_rh, 1, -1)

        lh_mask = lh_mask.astype(bool)
        rh_mask = rh_mask.astype(bool)

        arm_mask  = np.zeros((height, width), dtype=np.uint8)
        thickness = max(int(knee_width * 0.45), cfg["settings"]["min_arm_thickness"])
        thickness = min(thickness, cfg["settings"]["max_arm_thickness"])

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

        red_layer        = np.zeros((height, width, 3), dtype=np.uint8)
        green_layer      = np.zeros((height, width, 3), dtype=np.uint8)
        blue_layer       = np.zeros((height, width, 3), dtype=np.uint8)
        dark_green_layer = np.zeros((height, width, 3), dtype=np.uint8)

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
    yellow_layer  = np.zeros((height, width, 3), dtype=np.uint8)
    yellow_layer[activity_mask] = [0, 255, 255]
    overlay = cv2.addWeighted(overlay, 1.0, yellow_layer, alpha, 0)

    return overlay

# Two separate detector options since IMAGE and VIDEO modes can't be mixed
image_pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
    num_poses=cfg["settings"]["num_poses"],
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.IMAGE)

video_pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
    num_poses=cfg["settings"]["num_poses"],
    output_segmentation_masks=False,
    running_mode=VisionRunningMode.VIDEO)

landmarks_rows = []

for video_path in video_files:
    print(f"\nProcessing: {video_path.stem}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, raw_first_frame = cap.read()
    if not ret:
        print(f"Could not read {video_path.stem}, skipping.")
        cap.release()
        continue

    # Resize and center onto 1680x1050 canvas
    first_frame = resize_to_canvas(raw_first_frame, TARGET_W, TARGET_H)
    height, width = first_frame.shape[:2]  # will always be TARGET_H, TARGET_W

    # --- FIRST FRAME: segmented reference image ---
    image_detector = PoseLandmarker.create_from_options(image_pose_options)
    overlay = process_first_frame(first_frame, image_detector, width, height)
    image_detector.close()

    if overlay is None:
        print(f"Could not segment first frame of {video_path.stem}, skipping.")
        cap.release()
        continue

    output_path = OUTPUT_FOLDER / f"{video_path.stem}_segmented.jpg"
    cv2.imwrite(str(output_path), overlay)
    print(f"Saved segmented image to: {output_path}")

    # --- FULL VIDEO LOOP: per-frame head circles at each dilation level ---
    video_detector = PoseLandmarker.create_from_options(video_pose_options)

    frame_rows = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_index = 0

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break

        # Resize and center each frame onto 1680x1050 canvas
        frame = resize_to_canvas(raw_frame, TARGET_W, TARGET_H)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_index / fps) * 1000)
        pose_result  = video_detector.detect_for_video(mp_image, timestamp_ms)

        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) == 2:
            sorted_poses = sorted(
                pose_result.pose_landmarks,
                key=lambda p: get_center_x(p)
            )
            left_pose, right_pose = sorted_poses

            for dilation in DILATION_LEVELS:
                lx, ly, lr = get_head_circle(left_pose,  width, height, dilation)
                rx, ry, rr = get_head_circle(right_pose, width, height, dilation)

                frame_rows.append({
                    "frame":             frame_index,
                    "dilation":          dilation,
                    "left_head_x":       lx,
                    "left_head_y":       ly,
                    "left_head_radius":  lr,
                    "right_head_x":      rx,
                    "right_head_y":      ry,
                    "right_head_radius": rr,
                })
        else:
            # No detection — one blank row per dilation level to keep things consistent
            for dilation in DILATION_LEVELS:
                frame_rows.append({
                    "frame":             frame_index,
                    "dilation":          dilation,
                    "left_head_x":       "",
                    "left_head_y":       "",
                    "left_head_radius":  "",
                    "right_head_x":      "",
                    "right_head_y":      "",
                    "right_head_radius": "",
                })

        frame_index += 1

        if frame_index % 100 == 0:
            percent = (frame_index / total_frames) * 100
            print(f"  {percent:.1f}% complete")

    cap.release()
    video_detector.close()

    # Save per-frame CSV
    frames_path = FRAMES_FOLDER / f"{video_path.stem}_frames.csv"
    with open(frames_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "dilation",
            "left_head_x", "left_head_y", "left_head_radius",
            "right_head_x", "right_head_y", "right_head_radius"
        ])
        writer.writeheader()
        writer.writerows(frame_rows)
    print(f"Saved frame data to: {frames_path}")

    # Use first detected frame for summary head centers
    first_detected = next((r for r in frame_rows if r["left_head_x"] != ""), None)
    if first_detected:
        landmarks_rows.append({
            "video":        video_path.stem,
            "left_head_x":  first_detected["left_head_x"],
            "left_head_y":  first_detected["left_head_y"],
            "right_head_x": first_detected["right_head_x"],
            "right_head_y": first_detected["right_head_y"],
        })

    print(f"Done with {video_path.stem}.")

# Save summary head centers CSV
landmarks_path = LANDMARKS_FOLDER / "head_centers.csv"
with open(landmarks_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "video", "left_head_x", "left_head_y", "right_head_x", "right_head_y"
    ])
    writer.writeheader()
    writer.writerows(landmarks_rows)

print(f"\nSaved head centers to: {landmarks_path}")
print("All done.")