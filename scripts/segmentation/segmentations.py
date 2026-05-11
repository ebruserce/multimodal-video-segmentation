import csv
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

CANVAS_W, CANVAS_H = 1680, 1050
CANVAS_BG   = (211, 211, 211)   # grey — canvas border outside video area
VIDEO_BG    = (0, 0, 0)         # black — video area not covered by any mask


def load_config():
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    local_cfg_path = Path("configs/paths.local.yaml")
    if local_cfg_path.exists():
        with open(local_cfg_path) as f:
            local = yaml.safe_load(f)
        cfg["paths"].update(local["paths"])
    return cfg


def get_canvas_offsets(vid_w, vid_h, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
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


def vid_to_canvas(x, y, cx1, cy1):
    return x + cx1, y + cy1


def compute_activity_region(df, vid_w, vid_h, cx1, cy1,
                             activity_dilation=1.0):
    """
    Compute a fixed activity region from the first detected frame.
    Returns (act_x, act_y, act_w, act_h) in canvas space.
    Mirrors the logic in segment_frames.py.
    """
    detected = df[df["detected"] == 1]
    if detected.empty:
        return None

    row = detected.iloc[0]

    def get_px(label, lm_idx):
        vx = row[f"{label}_lm{lm_idx}_x"]
        vy = row[f"{label}_lm{lm_idx}_y"]
        return vid_to_canvas(vx, vy, cx1, cy1)

    # Horizontal: left elbow of a1 to right elbow of a2
    # MediaPipe indices: left elbow=13, right elbow=14
    a1_lelbow = get_px("a1", 13)
    a2_relbow = get_px("a2", 14)

    # Vertical: mid-torso to feet
    # shoulder=11/12, hip=23/24, foot=31/32
    a1_ls   = get_px("a1", 11)
    a1_lhip = get_px("a1", 23)
    a2_rs   = get_px("a2", 12)
    a2_rhip = get_px("a2", 24)
    a1_foot = get_px("a1", 31)
    a2_foot = get_px("a2", 32)

    act_x1 = int(a1_lelbow[0])
    act_x2 = int(a2_relbow[0])
    act_y1 = int(min((a1_ls[1] + a1_lhip[1]) / 2,
                     (a2_rs[1] + a2_rhip[1]) / 2))
    act_y2 = int(max(a1_foot[1], a2_foot[1]))

    # Apply dilation to width from center
    act_cx     = (act_x1 + act_x2) // 2
    act_half_w = int((act_x2 - act_x1) / 2 * activity_dilation)
    act_x1 = max(0, act_cx - act_half_w)
    act_x2 = min(CANVAS_W, act_cx + act_half_w)
    act_y1 = max(0, act_y1)
    act_y2 = min(CANVAS_H, act_y2)

    return (act_x1, act_y1, act_x2 - act_x1, act_y2 - act_y1)


def compute_segmentations(pose_csv_path, vid_w, vid_h,
                           head_dilation=0.75,
                           hand_dilation=2.0,
                           arm_dilation=0.45,
                           activity_dilation=1.0,
                           min_head_radius=20,
                           max_head_radius=200,
                           min_hand_radius=20,
                           max_hand_radius=150,
                           min_arm_thickness=15,
                           max_arm_thickness=100):
    """
    Reads pose CSV and computes segmentation parameters per frame.
    All output coordinates are in canvas (1680x1050) space.
    Activity region is fixed from the first detected frame.
    Returns list of dicts, one per frame.
    """
    cx1, cx2, cy1, cy2, vx1, vx2, vy1, vy2 = get_canvas_offsets(vid_w, vid_h)
    df = pd.read_csv(pose_csv_path)

    # Compute fixed activity region once
    activity = compute_activity_region(
        df, vid_w, vid_h, cx1, cy1, activity_dilation
    )

    rows = []

    for _, row in df.iterrows():
        frame = int(row["frame"])
        seg = {
            "frame":        frame,
            "timestamp_ms": int(row["timestamp_ms"]),
        }

        # Always write fixed activity region
        if activity:
            seg["activity_x"] = activity[0]
            seg["activity_y"] = activity[1]
            seg["activity_w"] = activity[2]
            seg["activity_h"] = activity[3]
        else:
            seg["activity_x"] = ""
            seg["activity_y"] = ""
            seg["activity_w"] = ""
            seg["activity_h"] = ""

        if not row["detected"]:
            for prefix in ["a1", "a2"]:
                seg[f"{prefix}_head_cx"]       = ""
                seg[f"{prefix}_head_cy"]       = ""
                seg[f"{prefix}_head_radius"]   = ""
                seg[f"{prefix}_lhand_cx"]      = ""
                seg[f"{prefix}_lhand_cy"]      = ""
                seg[f"{prefix}_lhand_radius"]  = ""
                seg[f"{prefix}_rhand_cx"]      = ""
                seg[f"{prefix}_rhand_cy"]      = ""
                seg[f"{prefix}_rhand_radius"]  = ""
                seg[f"{prefix}_larm_x1"]       = ""
                seg[f"{prefix}_larm_y1"]       = ""
                seg[f"{prefix}_larm_x2"]       = ""
                seg[f"{prefix}_larm_y2"]       = ""
                seg[f"{prefix}_larm_x3"]       = ""
                seg[f"{prefix}_larm_y3"]       = ""
                seg[f"{prefix}_rarm_x1"]       = ""
                seg[f"{prefix}_rarm_y1"]       = ""
                seg[f"{prefix}_rarm_x2"]       = ""
                seg[f"{prefix}_rarm_y2"]       = ""
                seg[f"{prefix}_rarm_x3"]       = ""
                seg[f"{prefix}_rarm_y3"]       = ""
                seg[f"{prefix}_arm_thickness"] = ""
            rows.append(seg)
            continue

        def get_px(label, lm_idx):
            vx = row[f"{label}_lm{lm_idx}_x"]
            vy = row[f"{label}_lm{lm_idx}_y"]
            return vid_to_canvas(vx, vy, cx1, cy1)

        for label in ["a1", "a2"]:
            # HEAD: y = mean of head landmarks (0-10), x = mean of shoulders (11,12)
            head_lm_indices = list(range(0, 11))
            head_pts = [get_px(label, i) for i in head_lm_indices]
            ls_x, ls_y = get_px(label, 11)
            rs_x, rs_y = get_px(label, 12)
            shoulder_w = abs(ls_x - rs_x)

            head_lm_cx = int(np.mean([p[0] for p in head_pts]))     # x = mean of head landmark points
            shoulder_cx = int((ls_x + rs_x) / 2)                    # x = shoulder midpoint
            head_cx = int((head_lm_cx + shoulder_cx) / 2)           # x = average of both
            head_cy = int(np.mean([p[1] for p in head_pts]))         # y = mean of head points

            head_r = int(shoulder_w * head_dilation)
            head_r = max(min_head_radius, min(head_r, max_head_radius))

            seg[f"{label}_head_cx"]     = head_cx
            seg[f"{label}_head_cy"]     = head_cy
            seg[f"{label}_head_radius"] = head_r

            # HANDS
            lh_pts = [get_px(label, i) for i in [17, 19, 21]]
            rh_pts = [get_px(label, i) for i in [18, 20, 22]]

            lh_w = np.linalg.norm(np.array(lh_pts[0]) - np.array(lh_pts[-1]))
            rh_w = np.linalg.norm(np.array(rh_pts[0]) - np.array(rh_pts[-1]))

            lh_cx = int(np.mean([p[0] for p in lh_pts]))
            lh_cy = int(np.mean([p[1] for p in lh_pts]))
            rh_cx = int(np.mean([p[0] for p in rh_pts]))
            rh_cy = int(np.mean([p[1] for p in rh_pts]))

            lh_r = max(min_hand_radius, min(int(lh_w * hand_dilation), max_hand_radius))
            rh_r = max(min_hand_radius, min(int(rh_w * hand_dilation), max_hand_radius))

            seg[f"{label}_lhand_cx"]     = lh_cx
            seg[f"{label}_lhand_cy"]     = lh_cy
            seg[f"{label}_lhand_radius"] = lh_r
            seg[f"{label}_rhand_cx"]     = rh_cx
            seg[f"{label}_rhand_cy"]     = rh_cy
            seg[f"{label}_rhand_radius"] = rh_r

            # ARMS
            l_shoulder = get_px(label, 11)
            l_elbow    = get_px(label, 13)
            l_wrist    = get_px(label, 15)
            r_shoulder = get_px(label, 12)
            r_elbow    = get_px(label, 14)
            r_wrist    = get_px(label, 16)

            thickness = int(shoulder_w * arm_dilation)
            thickness = max(min_arm_thickness, min(thickness, max_arm_thickness))

            seg[f"{label}_larm_x1"]       = int(l_shoulder[0])
            seg[f"{label}_larm_y1"]       = int(l_shoulder[1])
            seg[f"{label}_larm_x2"]       = int(l_elbow[0])
            seg[f"{label}_larm_y2"]       = int(l_elbow[1])
            seg[f"{label}_larm_x3"]       = int(l_wrist[0])
            seg[f"{label}_larm_y3"]       = int(l_wrist[1])
            seg[f"{label}_rarm_x1"]       = int(r_shoulder[0])
            seg[f"{label}_rarm_y1"]       = int(r_shoulder[1])
            seg[f"{label}_rarm_x2"]       = int(r_elbow[0])
            seg[f"{label}_rarm_y2"]       = int(r_elbow[1])
            seg[f"{label}_rarm_x3"]       = int(r_wrist[0])
            seg[f"{label}_rarm_y3"]       = int(r_wrist[1])
            seg[f"{label}_arm_thickness"] = thickness

        rows.append(seg)

    return rows


def save_segmentations(rows, output_path):
    if not rows:
        print("No rows to save.")
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved segmentation data to: {output_path}")


def render_segmentation_video(video_path, seg_csv_path, mask_npy_path,
                               output_path,
                               video_alpha=0.0,
                               canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """
    Renders segmentation video with colored regions.
    video_alpha: 0.0 = black video bg, 1.0 = original video fully visible.
    Layer priority (lowest to highest): body, arms, hands, head, activity.
    """
    seg_df = pd.read_csv(seg_csv_path)
    masks  = np.load(str(mask_npy_path))   # shape: (total_frames, vid_h, vid_w)

    cap = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cx1, cx2, cy1, cy2, vx1, vx2, vy1, vy2 = get_canvas_offsets(
        vid_w, vid_h, canvas_w, canvas_h
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Canvas: grey border, black video area
        canvas = np.full((canvas_h, canvas_w, 3), CANVAS_BG, dtype=np.uint8)
        canvas[cy1:cy2, cx1:cx2] = VIDEO_BG

        # Optionally blend original video underneath
        if video_alpha > 0:
            canvas[cy1:cy2, cx1:cx2] = (
                np.array(VIDEO_BG, dtype=np.float32) * (1 - video_alpha) +
                frame[vy1:vy2, vx1:vx2].astype(np.float32) * video_alpha
            ).astype(np.uint8)

        seg_rows = seg_df[seg_df["frame"] == frame_index]
        if len(seg_rows) == 0:
            out.write(canvas)
            frame_index += 1
            continue

        seg = seg_rows.iloc[0]

        # --- BODY MASK (lowest priority) ---
        if frame_index < len(masks):
            body_mask_vid = masks[frame_index]                         # (vid_h, vid_w)
            body_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
            body_canvas[cy1:cy2, cx1:cx2] = body_mask_vid[vy1:vy2, vx1:vx2]
            canvas[body_canvas] = [255, 0, 0]                         # blue

        # Draw order: arms -> hands -> head -> activity (each overwrites lower priority)
        for label in ["a1", "a2"]:
            if seg[f"{label}_head_cx"] == "" or pd.isna(seg[f"{label}_head_cx"]):
                continue

            t = int(seg[f"{label}_arm_thickness"])

            # ARMS (green)
            cv2.line(canvas,
                     (int(seg[f"{label}_larm_x1"]), int(seg[f"{label}_larm_y1"])),
                     (int(seg[f"{label}_larm_x2"]), int(seg[f"{label}_larm_y2"])),
                     (0, 255, 0), t)
            cv2.line(canvas,
                     (int(seg[f"{label}_larm_x2"]), int(seg[f"{label}_larm_y2"])),
                     (int(seg[f"{label}_larm_x3"]), int(seg[f"{label}_larm_y3"])),
                     (0, 255, 0), t)
            cv2.line(canvas,
                     (int(seg[f"{label}_rarm_x1"]), int(seg[f"{label}_rarm_y1"])),
                     (int(seg[f"{label}_rarm_x2"]), int(seg[f"{label}_rarm_y2"])),
                     (0, 255, 0), t)
            cv2.line(canvas,
                     (int(seg[f"{label}_rarm_x2"]), int(seg[f"{label}_rarm_y2"])),
                     (int(seg[f"{label}_rarm_x3"]), int(seg[f"{label}_rarm_y3"])),
                     (0, 255, 0), t)

            # HANDS (dark green)
            cv2.circle(canvas,
                       (int(seg[f"{label}_lhand_cx"]), int(seg[f"{label}_lhand_cy"])),
                       int(seg[f"{label}_lhand_radius"]),
                       (0, 128, 0), -1)
            cv2.circle(canvas,
                       (int(seg[f"{label}_rhand_cx"]), int(seg[f"{label}_rhand_cy"])),
                       int(seg[f"{label}_rhand_radius"]),
                       (0, 128, 0), -1)

            # HEAD (red)
            cv2.circle(canvas,
                       (int(seg[f"{label}_head_cx"]), int(seg[f"{label}_head_cy"])),
                       int(seg[f"{label}_head_radius"]),
                       (0, 0, 255), -1)

        # ACTIVITY (yellow — drawn last, highest priority)
        if seg["activity_x"] != "" and not pd.isna(seg["activity_x"]):
            ax = int(seg["activity_x"])
            ay = int(seg["activity_y"])
            aw = int(seg["activity_w"])
            ah = int(seg["activity_h"])
            cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), -1)

        out.write(canvas)
        frame_index += 1

        if frame_index % 100 == 0:
            pct = (frame_index / total_frames) * 100
            print(f"  {pct:.1f}% complete")

    cap.release()
    out.release()
    print(f"Saved segmentation video to: {output_path}")


if __name__ == "__main__":
    cfg = load_config()
    ROOT             = Path(cfg["paths"].get("project_root", "."))
    video_folder     = ROOT / cfg["paths"]["data"]["input_videos"]
    landmarks_folder = ROOT / cfg["paths"]["data"]["landmarks"]
    frames_folder    = landmarks_folder / "frames"
    output_folder    = ROOT / cfg["paths"]["data"]["output"]
    output_folder.mkdir(parents=True, exist_ok=True)

    video_files = list(video_folder.glob(cfg["settings"]["video_input_extension"]))
    if not video_files:
        print("No video files found.")
        exit(1)

    video_path = video_files[0]
    cap = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    pose_csv  = frames_folder / f"{video_path.stem}_pose.csv"
    mask_npy  = frames_folder / f"{video_path.stem}_masks.npy"
    seg_csv   = frames_folder / f"{video_path.stem}_segmentations.csv"
    seg_video = output_folder / f"{video_path.stem}_segmentation.mp4"

    rows = compute_segmentations(
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
    save_segmentations(rows, seg_csv)
    render_segmentation_video(
        video_path, seg_csv, mask_npy, seg_video,
        video_alpha=cfg["settings"].get("seg_video_alpha", 0.0)
    )
    print("Done.")