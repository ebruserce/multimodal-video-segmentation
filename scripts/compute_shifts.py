import csv
import yaml
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
LANDMARKS_FOLDER = ROOT / cfg["paths"]["data"]["landmarks"]

reference_video = cfg["settings"]["reference_video"]

# Load head centers CSV
head_centers_path = LANDMARKS_FOLDER / "head_centers.csv"
if not head_centers_path.exists():
    print(f"Could not find head_centers.csv at {head_centers_path}")
    exit(1)

with open(head_centers_path, newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Find reference row
ref_row = next((r for r in rows if r["video"] == reference_video), None)
if ref_row is None:
    print(f"Reference video '{reference_video}' not found in head_centers.csv")
    print(f"Available videos: {[r['video'] for r in rows]}")
    exit(1)

ref_left_x  = int(ref_row["left_head_x"])
ref_left_y  = int(ref_row["left_head_y"])
ref_right_x = int(ref_row["right_head_x"])
ref_right_y = int(ref_row["right_head_y"])

print(f"Reference video: {reference_video}")
print(f"  Left head:  ({ref_left_x}, {ref_left_y})")
print(f"  Right head: ({ref_right_x}, {ref_right_y})")

# Compute shifts for all videos (reference will have 0, 0, 0, 0)
shifts = []
for row in rows:
    left_shift_x  = ref_left_x  - int(row["left_head_x"])
    left_shift_y  = ref_left_y  - int(row["left_head_y"])
    right_shift_x = ref_right_x - int(row["right_head_x"])
    right_shift_y = ref_right_y - int(row["right_head_y"])

    shifts.append({
        "video":         row["video"],
        "left_shift_x":  left_shift_x,
        "left_shift_y":  left_shift_y,
        "right_shift_x": right_shift_x,
        "right_shift_y": right_shift_y,
    })

    print(f"{row['video']}: left shift ({left_shift_x}, {left_shift_y}), right shift ({right_shift_x}, {right_shift_y})")

# Save shifts CSV
shifts_path = LANDMARKS_FOLDER / "shifts.csv"
with open(shifts_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "video", "left_shift_x", "left_shift_y", "right_shift_x", "right_shift_y"
    ])
    writer.writeheader()
    writer.writerows(shifts)

print(f"\nSaved shifts to: {shifts_path}")