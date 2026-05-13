# Ebru Serce, 2026
import csv
import yaml
import cv2
import pandas as pd
import numpy as np
from pathlib import Path


def load_config():
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    local_cfg_path = Path("configs/paths.local.yaml")
    if local_cfg_path.exists():
        with open(local_cfg_path) as f:
            local = yaml.safe_load(f)
        cfg["paths"].update(local["paths"])
    return cfg

def point_in_circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

def point_in_rect(x, y, rx, ry, rw, rh):
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def point_in_thick_line(x, y, x1, y1, x2, y2, thickness):
    """Check if point is within thickness/2 of a line segment."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return (x - x1) ** 2 + (y - y1) ** 2 <= (thickness / 2) ** 2
    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    dist_sq = (x - proj_x) ** 2 + (y - proj_y) ** 2
    return dist_sq <= (thickness / 2) ** 2

CANVAS_W, CANVAS_H = 1680, 1050

def classify_gaze_point(x, y, seg, body_mask_canvas,
                         canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """
    Classify a single gaze point against segmentation regions.
    Priority: screen > activity > head > hands > arms > body > bg
    body_mask_canvas: 2D bool array (canvas_h, canvas_w) for this frame,
                      or None if not available.
    """
    vid_x1 = seg.get("vid_cx1", 0)
    vid_x2 = seg.get("vid_cx2", canvas_w)
    vid_y1 = seg.get("vid_cy1", 0)
    vid_y2 = seg.get("vid_cy2", canvas_h)

    ix, iy = int(x), int(y)

    # am_screen: outside original video area
    if not (vid_x1 <= x <= vid_x2 and vid_y1 <= y <= vid_y2):
        return "am_screen"

    # am_activity
    if seg["activity_x"] != "" and point_in_rect(
        x, y,
        int(seg["activity_x"]), int(seg["activity_y"]),
        int(seg["activity_w"]), int(seg["activity_h"])
    ):
        return "am_activity"

    # HEAD
    for label, aoi in [("a1", "am_a1head"), ("a2", "am_a2head")]:
        if seg[f"{label}_head_cx"] != "" and point_in_circle(
            x, y,
            int(seg[f"{label}_head_cx"]),
            int(seg[f"{label}_head_cy"]),
            int(seg[f"{label}_head_radius"])
        ):
            return aoi

    # HANDS
    for label, aoi in [("a1", "am_a1hands"), ("a2", "am_a2hands")]:
        if seg[f"{label}_lhand_cx"] != "":
            if point_in_circle(x, y,
                               int(seg[f"{label}_lhand_cx"]),
                               int(seg[f"{label}_lhand_cy"]),
                               int(seg[f"{label}_lhand_radius"])):
                return aoi
            if point_in_circle(x, y,
                               int(seg[f"{label}_rhand_cx"]),
                               int(seg[f"{label}_rhand_cy"]),
                               int(seg[f"{label}_rhand_radius"])):
                return aoi

    # ARMS
    for label, aoi in [("a1", "am_a1arms"), ("a2", "am_a2arms")]:
        if seg[f"{label}_larm_x1"] != "":
            t = int(seg[f"{label}_arm_thickness"])
            if point_in_thick_line(x, y,
                                   int(seg[f"{label}_larm_x1"]), int(seg[f"{label}_larm_y1"]),
                                   int(seg[f"{label}_larm_x2"]), int(seg[f"{label}_larm_y2"]), t):
                return aoi
            if point_in_thick_line(x, y,
                                   int(seg[f"{label}_larm_x2"]), int(seg[f"{label}_larm_y2"]),
                                   int(seg[f"{label}_larm_x3"]), int(seg[f"{label}_larm_y3"]), t):
                return aoi
            if point_in_thick_line(x, y,
                                   int(seg[f"{label}_rarm_x1"]), int(seg[f"{label}_rarm_y1"]),
                                   int(seg[f"{label}_rarm_x2"]), int(seg[f"{label}_rarm_y2"]), t):
                return aoi
            if point_in_thick_line(x, y,
                                   int(seg[f"{label}_rarm_x2"]), int(seg[f"{label}_rarm_y2"]),
                                   int(seg[f"{label}_rarm_x3"]), int(seg[f"{label}_rarm_y3"]), t):
                return aoi

    # BODY — use segmentation mask, split left/right by canvas center
    if body_mask_canvas is not None:
        if 0 <= iy < canvas_h and 0 <= ix < canvas_w:
            if body_mask_canvas[iy, ix]:
                # Determine which person based on canvas x midpoint
                canvas_mid_x = canvas_w // 2
                if x < canvas_mid_x:
                    return "am_a1body"
                else:
                    return "am_a2body"

    return "am_bg"


def compute_gaze_locations(gaze_csv_path, seg_csv_path, mask_npy_path,
                            vid_w, vid_h,
                            participant_id,
                            video_name,
                            canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """
    For each valid gaze point, determine which segmentation region it falls in.
    Returns one row per gaze point with participant_id, video_name, frame, t, sx, sy, predicted_aoi.
    """
    gaze_df = pd.read_csv(gaze_csv_path)
    seg_df  = pd.read_csv(seg_csv_path)

    masks = np.load(str(mask_npy_path), mmap_mode='r')

    cx1 = (canvas_w - vid_w) // 2
    cx2 = cx1 + vid_w
    cy1 = (canvas_h - vid_h) // 2
    cy2 = cy1 + vid_h

    seg_by_ts = {
        int(row["timestamp_ms"]): row.to_dict()
        for _, row in seg_df.iterrows()
    }
    all_timestamps = sorted(seg_by_ts.keys())

    frame_by_ts = {
        int(row["timestamp_ms"]): int(row["frame"])
        for _, row in seg_df.iterrows()
    }

    results = []

    for _, gaze_row in gaze_df.iterrows():
        t     = int(gaze_row["t"])
        sx    = gaze_row["sx"]
        sy    = gaze_row["sy"]
        valid = int(gaze_row["valid"])

        closest_ts = min(all_timestamps, key=lambda ts: abs(ts - t))
        frame_idx = frame_by_ts.get(closest_ts)

        base_row = {
            "video_name": video_name,
            "participant_id": participant_id,
            "frame": frame_idx,
            "t": t,
            "sx": sx,
            "sy": sy,
        }

        if not valid or pd.isna(sx) or pd.isna(sy):
            results.append({**base_row, "predicted_aoi": "invalid"})
            continue

        sx, sy = float(sx), float(sy)

        seg = seg_by_ts[closest_ts].copy()
        seg["vid_cx1"] = cx1
        seg["vid_cx2"] = cx2
        seg["vid_cy1"] = cy1
        seg["vid_cy2"] = cy2

        body_mask_canvas = None
        if frame_idx is not None and frame_idx < len(masks):
            body_mask_canvas = None
            if frame_idx is not None and frame_idx < len(masks):
                body_mask_vid = masks[frame_idx]
                body_mask_canvas = np.zeros((canvas_h, canvas_w), dtype=bool)

                vy1 = max(0, -((canvas_h - vid_h) // 2))
                vy2 = vy1 + (cy2 - cy1)
                vx1 = max(0, -((canvas_w - vid_w) // 2))
                vx2 = vx1 + (cx2 - cx1)

                body_mask_canvas[cy1:cy2, cx1:cx2] = body_mask_vid[vy1:vy2, vx1:vx2]
        predicted_aoi = classify_gaze_point(
            sx, sy, seg, body_mask_canvas, canvas_w, canvas_h
        )

        results.append({**base_row, "predicted_aoi": predicted_aoi})

    return results


def save_gaze_locations(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "participant_id",
                "frame",
                "t",
                "sx",
                "sy",
                "predicted_aoi",
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved gaze locations to: {output_path}")


if __name__ == "__main__":
    import re

    cfg = load_config()
    ROOT             = Path(cfg["paths"].get("project_root", "."))
    video_folder     = ROOT / cfg["paths"]["data"]["input_videos"]
    gaze_root        = ROOT / cfg["paths"]["data"]["gaze_folder"]
    landmarks_folder = ROOT / cfg["paths"]["data"]["landmarks"]
    frames_folder    = landmarks_folder / "frames"
    output_folder    = ROOT / cfg["paths"]["data"]["output"]

    gaze_output_root = output_folder / "gaze_locations"
    gaze_output_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(video_folder.glob(cfg["settings"]["video_input_extension"]))

    if not video_files:
        print("No video files found.")
        exit(1)

    def extract_id(fname):
        match = re.search(r'~~([^~]+)~~', fname)
        return match.group(1) if match else Path(fname).stem

    print(f"Found {len(video_files)} video(s).")

    for video_path in video_files:
        video_name = video_path.stem

        print(f"\n{'=' * 60}")
        print(f"Processing video: {video_name}")
        print(f"{'=' * 60}")

        video_gaze_folder = gaze_root / video_name

        if not video_gaze_folder.exists():
            print(f"Missing gaze folder for video, skipping: {video_gaze_folder}")
            continue

        gaze_files = sorted(video_gaze_folder.glob("*.csv"))

        if not gaze_files:
            print(f"No gaze files found in: {video_gaze_folder}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        seg_csv  = frames_folder / f"{video_name}_segmentations.csv"
        mask_npy = frames_folder / f"{video_name}_masks.npy"

        if not seg_csv.exists():
            print(f"Missing segmentation CSV, skipping: {seg_csv}")
            continue

        if not mask_npy.exists():
            print(f"Missing mask file, skipping: {mask_npy}")
            continue

        combined_results = []

        for gaze_path in gaze_files:
            participant_id = extract_id(gaze_path.name)

            print(f"  Processing participant: {participant_id}")

            results = compute_gaze_locations(
                gaze_csv_path=gaze_path,
                seg_csv_path=seg_csv,
                mask_npy_path=mask_npy,
                vid_w=vid_w,
                vid_h=vid_h,
                participant_id=participant_id,
                video_name=video_name,
            )

            combined_results.extend(results)

        video_output_folder = gaze_output_root / video_name
        out_path = video_output_folder / f"{video_name}_gaze_locations_all.csv"

        save_gaze_locations(combined_results, out_path)

    print("\nDone.")