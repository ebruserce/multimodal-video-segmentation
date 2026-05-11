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

def classify_gaze_point(x, y, seg, canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """
    Classify a single gaze point against segmentation regions.
    Priority: screen > activity > head > hands > arms > body > bg
    Returns region label string.
    """
    # am_screen: outside the original video area (gray borders)
    # We detect this by checking if the point is outside the video bounds
    # seg contains activity_x etc which implicitly defines canvas usage,
    # but we need vid bounds — passed via seg dict
    vid_x1 = seg.get("vid_cx1", 0)
    vid_x2 = seg.get("vid_cx2", canvas_w)
    vid_y1 = seg.get("vid_cy1", 0)
    vid_y2 = seg.get("vid_cy2", canvas_h)

    if not (vid_x1 <= x <= vid_x2 and vid_y1 <= y <= vid_y2):
        return "am_screen"

    # am_activity (takes priority over hands, arms, body)
    if seg["activity_x"] != "" and point_in_rect(
        x, y,
        int(seg["activity_x"]), int(seg["activity_y"]),
        int(seg["activity_w"]), int(seg["activity_h"])
    ):
        return "am_activity"

    # HEAD (highest priority among people regions)
    for label, aoi in [("a1", "am_a1head"), ("a2", "am_a2head")]:
        if seg[f"{label}_head_cx"] != "" and point_in_circle(
            x, y,
            int(seg[f"{label}_head_cx"]),
            int(seg[f"{label}_head_cy"]),
            int(seg[f"{label}_head_radius"])
        ):
            return aoi

    # HANDS
    for label, laoi, raoi in [("a1", "am_a1hands", "am_a1hands"),
                               ("a2", "am_a2hands", "am_a2hands")]:
        if seg[f"{label}_lhand_cx"] != "":
            if point_in_circle(x, y,
                               int(seg[f"{label}_lhand_cx"]),
                               int(seg[f"{label}_lhand_cy"]),
                               int(seg[f"{label}_lhand_radius"])):
                return laoi
            if point_in_circle(x, y,
                               int(seg[f"{label}_rhand_cx"]),
                               int(seg[f"{label}_rhand_cy"]),
                               int(seg[f"{label}_rhand_radius"])):
                return raoi

    # ARMS
    for label, aoi in [("a1", "am_a1arms"), ("a2", "am_a2arms")]:
        if seg[f"{label}_larm_x1"] != "":
            t = int(seg[f"{label}_arm_thickness"])
            # Left arm: shoulder->elbow, elbow->wrist
            if point_in_thick_line(x, y,
                                   int(seg[f"{label}_larm_x1"]), int(seg[f"{label}_larm_y1"]),
                                   int(seg[f"{label}_larm_x2"]), int(seg[f"{label}_larm_y2"]), t):
                return aoi
            if point_in_thick_line(x, y,
                                   int(seg[f"{label}_larm_x2"]), int(seg[f"{label}_larm_y2"]),
                                   int(seg[f"{label}_larm_x3"]), int(seg[f"{label}_larm_y3"]), t):
                return aoi
            # Right arm
            if point_in_thick_line(x, y,
                                   int(seg[f"{label}_rarm_x1"]), int(seg[f"{label}_rarm_y1"]),
                                   int(seg[f"{label}_rarm_x2"]), int(seg[f"{label}_rarm_y2"]), t):
                return aoi
            if point_in_thick_line(x, y,
                                   int(seg[f"{label}_rarm_x2"]), int(seg[f"{label}_rarm_y2"]),
                                   int(seg[f"{label}_rarm_x3"]), int(seg[f"{label}_rarm_y3"]), t):
                return aoi

    # BODY (inside video bounds but no other region matched)
    return "am_bg"

def compute_gaze_locations(gaze_csv_path, seg_csv_path, vid_w, vid_h,
                            canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """
    For each valid gaze point, determine which segmentation region it falls in.
    Returns list of dicts with t, sx, sy, predicted_aoi.
    """
    gaze_df = pd.read_csv(gaze_csv_path)
    seg_df  = pd.read_csv(seg_csv_path)

    # Compute video bounds on canvas for am_screen detection
    cx1 = (canvas_w - vid_w) // 2
    cx2 = cx1 + vid_w
    cy1 = (canvas_h - vid_h) // 2
    cy2 = cy1 + vid_h

    # Build seg lookup by timestamp_ms for fast access
    seg_by_ts = {int(row["timestamp_ms"]): row.to_dict()
                 for _, row in seg_df.iterrows()}
    all_timestamps = sorted(seg_by_ts.keys())

    results = []

    for _, gaze_row in gaze_df.iterrows():
        t   = int(gaze_row["t"])
        sx  = gaze_row["sx"]
        sy  = gaze_row["sy"]
        valid = int(gaze_row["valid"])

        if not valid or pd.isna(sx) or pd.isna(sy):
            results.append({"t": t, "sx": sx, "sy": sy, "predicted_aoi": "invalid"})
            continue

        sx, sy = float(sx), float(sy)

        # Find closest segmentation frame by timestamp
        closest_ts = min(all_timestamps, key=lambda ts: abs(ts - t))
        seg = seg_by_ts[closest_ts]
        seg["vid_cx1"] = cx1
        seg["vid_cx2"] = cx2
        seg["vid_cy1"] = cy1
        seg["vid_cy2"] = cy2

        predicted_aoi = classify_gaze_point(sx, sy, seg, canvas_w, canvas_h)
        results.append({"t": t, "sx": sx, "sy": sy, "predicted_aoi": predicted_aoi})

    return results

def save_gaze_locations(results, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "sx", "sy", "predicted_aoi"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved gaze locations to: {output_path}")


if __name__ == "__main__":
    import re
    cfg = load_config()
    ROOT = Path(cfg["paths"].get("project_root", "."))
    video_folder    = ROOT / cfg["paths"]["data"]["input_videos"]
    gaze_folder     = ROOT / cfg["paths"]["data"]["gaze_folder"]
    landmarks_folder = ROOT / cfg["paths"]["data"]["landmarks"]
    frames_folder   = landmarks_folder / "frames"
    output_folder   = ROOT / cfg["paths"]["data"]["output"]
    output_folder.mkdir(parents=True, exist_ok=True)

    video_files = list(video_folder.glob(cfg["settings"]["video_input_extension"]))
    gaze_files  = list(gaze_folder.glob("*.csv"))

    if not video_files:
        print("No video files found.")
        exit(1)
    if not gaze_files:
        print("No gaze files found.")
        exit(1)

    video_path = video_files[0]
    cap = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    seg_csv = frames_folder / f"{video_path.stem}_segmentations.csv"

    def extract_id(fname):
        match = re.search(r'~~([^~]+)~~', fname)
        return match.group(1) if match else fname

    for gaze_path in gaze_files:
        pid = extract_id(gaze_path.name)
        print(f"Processing gaze file: {pid}")
        results = compute_gaze_locations(gaze_path, seg_csv, vid_w, vid_h)
        out_path = output_folder / f"{video_path.stem}_{pid}_gaze_locations.csv"
        save_gaze_locations(results, out_path)

    print("Done.")