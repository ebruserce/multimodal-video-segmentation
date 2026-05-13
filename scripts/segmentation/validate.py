# Ebru Serce, 2026

import csv
import re
import yaml
import cv2
import numpy as np
import pandas as pd
import bisect
from pathlib import Path

from segmentations import compute_segmentations, save_segmentations
from gaze_locations import classify_gaze_point


HEAD_DILATIONS = [round(x * 0.1, 1) for x in range(1, 16)]  # 0.1 to 1.5


def load_config():
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)

    local_cfg_path = Path("configs/paths.local.yaml")
    if local_cfg_path.exists():
        with open(local_cfg_path) as f:
            local = yaml.safe_load(f)
        cfg["paths"].update(local["paths"])

    return cfg


AOI_NORMALIZE = {
    "am_distractors": "am_bg",
    "offscreen": "invalid",
    "invalid": "invalid",
}


def normalize_aoi(aoi):
    return AOI_NORMALIZE.get(aoi, aoi)


def extract_id(fname):
    match = re.search(r'~~([^~]+)~~', fname)
    return match.group(1) if match else Path(fname).stem


def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(str(video_path))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return vid_w, vid_h


def safe_pct(matches, total):
    if total == 0:
        return np.nan
    return round(100 * matches / total, 2)

def find_closest_timestamp(sorted_timestamps, t):
    pos = bisect.bisect_left(sorted_timestamps, t)

    if pos == 0:
        return sorted_timestamps[0]

    if pos == len(sorted_timestamps):
        return sorted_timestamps[-1]

    before = sorted_timestamps[pos - 1]
    after = sorted_timestamps[pos]

    if abs(after - t) < abs(t - before):
        return after

    return before

def compute_gaze_locations_no_body(gaze_csv_path, seg_csv_path,
                                   vid_w, vid_h,
                                   participant_id,
                                   video_name,
                                   canvas_w=1680, canvas_h=1050):
    """
    Faster validation-only gaze classification.
    Skips body-mask checking entirely.

    Gaze points that would have matched body will fall through to am_bg
    unless they match a higher-priority AOI first.
    """
    gaze_df = pd.read_csv(gaze_csv_path)
    seg_df = pd.read_csv(seg_csv_path)

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
        t = int(gaze_row["t"])
        sx = gaze_row["sx"]
        sy = gaze_row["sy"]
        valid = int(gaze_row["valid"])

        closest_ts = find_closest_timestamp(all_timestamps, t)
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

        predicted_aoi = classify_gaze_point(
            sx,
            sy,
            seg,
            body_mask_canvas=None,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )

        results.append({**base_row, "predicted_aoi": predicted_aoi})

    return results

def compute_validation_from_results(results, gaze_csv_path):
    """
    Compare predicted AOIs against the original aoi column for one participant.
    Returns one summary dict.
    """
    pred_df = pd.DataFrame(results)
    truth_df = pd.read_csv(gaze_csv_path)

    if "aoi" not in truth_df.columns:
        print(f"  Warning: missing aoi column in {gaze_csv_path}")
        return None

    merged = pd.merge(
        pred_df,
        truth_df[["t", "aoi"]],
        on="t",
        how="inner"
    )

    if len(merged) == 0:
        return None

    merged["aoi_normalized"] = merged["aoi"].apply(normalize_aoi)
    merged["match"] = merged["predicted_aoi"] == merged["aoi_normalized"]

    def region_accuracy(region):
        rows = merged[merged["aoi_normalized"] == region]
        return safe_pct(rows["match"].sum(), len(rows)), len(rows)

    overall_pct = safe_pct(merged["match"].sum(), len(merged))

    a1head_pct, a1head_total = region_accuracy("am_a1head")
    a2head_pct, a2head_total = region_accuracy("am_a2head")
    activity_pct, activity_total = region_accuracy("am_activity")
    bg_pct, bg_total = region_accuracy("am_bg")
    body1_pct, body1_total = region_accuracy("am_a1body")
    body2_pct, body2_total = region_accuracy("am_a2body")
    hands1_pct, hands1_total = region_accuracy("am_a1hands")
    hands2_pct, hands2_total = region_accuracy("am_a2hands")
    arms1_pct, arms1_total = region_accuracy("am_a1arms")
    arms2_pct, arms2_total = region_accuracy("am_a2arms")

    head_rows = merged[merged["aoi_normalized"].isin(["am_a1head", "am_a2head"])]
    total_head_pct = safe_pct(head_rows["match"].sum(), len(head_rows))

    return {
        "overall_pct": overall_pct,

        "total_head_pct": total_head_pct,
        "a1head_pct": a1head_pct,
        "a2head_pct": a2head_pct,
        "activity_pct": activity_pct,
        "bg_pct": bg_pct,

        "a1body_pct": body1_pct,
        "a2body_pct": body2_pct,
        "a1hands_pct": hands1_pct,
        "a2hands_pct": hands2_pct,
        "a1arms_pct": arms1_pct,
        "a2arms_pct": arms2_pct,

        "overall_n": len(merged),
        "total_head_n": len(head_rows),
        "a1head_n": a1head_total,
        "a2head_n": a2head_total,
        "activity_n": activity_total,
        "bg_n": bg_total,
    }


def mean_ignore_missing(values):
    values = [v for v in values if not pd.isna(v)]
    if len(values) == 0:
        return np.nan
    return round(float(np.mean(values)), 2)


def summarize_dilation(video_name, dilation, participant_results):
    """
    Average participant-level validation results.
    Missing region values are ignored.
    """
    row = {
        "video_name": video_name,
        "head_dilation": dilation,
        "num_participants": len(participant_results),
    }

    metric_cols = [
        "overall_pct",
        "total_head_pct",
        "a1head_pct",
        "a2head_pct",
        "activity_pct",
        "bg_pct",
        "a1body_pct",
        "a2body_pct",
        "a1hands_pct",
        "a2hands_pct",
        "a1arms_pct",
        "a2arms_pct",
    ]

    for col in metric_cols:
        row[f"avg_{col}"] = mean_ignore_missing(
            [r[col] for r in participant_results if r is not None]
        )

    count_cols = [
        "overall_n",
        "total_head_n",
        "a1head_n",
        "a2head_n",
        "activity_n",
        "bg_n",
    ]

    for col in count_cols:
        row[col] = int(sum(
            0 if pd.isna(r.get(col, np.nan)) else r.get(col, 0)
            for r in participant_results
            if r is not None
        ))

    return row


def choose_best_dilation(summary_rows):
    """
    Choose the dilation with the highest average overall accuracy.
    """
    def score(row):
        return -1 if pd.isna(row["avg_overall_pct"]) else row["avg_overall_pct"]

    return max(summary_rows, key=score)


def save_rows(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to save for {output_path}")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


def main():
    cfg = load_config()

    ROOT = Path(cfg["paths"].get("project_root", "."))

    video_folder = ROOT / cfg["paths"]["data"]["input_videos"]
    gaze_root = ROOT / cfg["paths"]["data"]["gaze_folder"]
    landmarks_folder = ROOT / cfg["paths"]["data"]["landmarks"]
    frames_folder = landmarks_folder / "frames"
    output_folder = ROOT / cfg["paths"]["data"]["output"]

    validation_root = output_folder / "dilation_validation"
    seg_validation_root = validation_root / "segmentations"

    video_files = sorted(video_folder.glob(cfg["settings"]["video_input_extension"]))

    if not video_files:
        print("No video files found.")
        return

    recommendation_rows = []

    for video_path in video_files:
        video_name = video_path.stem

        print(f"\n{'=' * 70}")
        print(f"Validating dilation levels for video: {video_name}")
        print(f"{'=' * 70}")

        video_gaze_folder = gaze_root / video_name

        if not video_gaze_folder.exists():
            print(f"Missing gaze folder, skipping: {video_gaze_folder}")
            continue

        gaze_files = sorted(video_gaze_folder.glob("*.csv"))

        if not gaze_files:
            print(f"No gaze files found, skipping: {video_gaze_folder}")
            continue

        pose_csv = frames_folder / f"{video_name}_pose.csv"
        mask_npy = frames_folder / f"{video_name}_masks.npy"

        if not pose_csv.exists():
            print(f"Missing pose CSV, skipping: {pose_csv}")
            continue

        if not mask_npy.exists():
            print(f"Missing mask file, skipping: {mask_npy}")
            continue

        vid_w, vid_h = get_video_dimensions(video_path)

        video_summary_rows = []

        for dilation in HEAD_DILATIONS:
            dil_str = str(dilation).replace(".", "p")

            print(f"\nTesting head dilation: {dilation}")

            seg_csv = (
                seg_validation_root
                / video_name
                / f"{video_name}_segmentations_hd{dil_str}.csv"
            )

            if not seg_csv.exists():
                seg_rows = compute_segmentations(
                    pose_csv_path=pose_csv,
                    vid_w=vid_w,
                    vid_h=vid_h,
                    head_dilation=dilation,
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
                seg_csv.parent.mkdir(parents=True, exist_ok=True)
                save_segmentations(seg_rows, seg_csv)

            participant_results = []

            for gaze_path in gaze_files:
                participant_id = extract_id(gaze_path.name)

                results = compute_gaze_locations_no_body(
                    gaze_csv_path=gaze_path,
                    seg_csv_path=seg_csv,
                    vid_w=vid_w,
                    vid_h=vid_h,
                    participant_id=participant_id,
                    video_name=video_name,
                )

                validation = compute_validation_from_results(results, gaze_path)

                if validation is not None:
                    participant_results.append(validation)

            if not participant_results:
                print(f"  No valid participant results for dilation {dilation}")
                continue

            summary_row = summarize_dilation(
                video_name=video_name,
                dilation=dilation,
                participant_results=participant_results,
            )

            video_summary_rows.append(summary_row)

            print(
                f"  avg_total_head={summary_row['avg_total_head_pct']}%, "
                f"avg_activity={summary_row['avg_activity_pct']}%, "
                f"avg_bg={summary_row['avg_bg_pct']}%, "
                f"avg_overall={summary_row['avg_overall_pct']}%"
            )

        if not video_summary_rows:
            print(f"No dilation summaries generated for {video_name}")
            continue

        video_summary_path = (
            validation_root
            / video_name
            / f"{video_name}_dilation_validation_summary.csv"
        )
        save_rows(video_summary_rows, video_summary_path)

        best = choose_best_dilation(video_summary_rows)

        recommendation_rows.append({
            "video_name": video_name,
            "recommended_head_dilation": best["head_dilation"],
            "avg_total_head_pct": best["avg_total_head_pct"],
            "avg_a1head_pct": best["avg_a1head_pct"],
            "avg_a2head_pct": best["avg_a2head_pct"],
            "avg_activity_pct": best["avg_activity_pct"],
            "avg_bg_pct": best["avg_bg_pct"],
            "avg_overall_pct": best["avg_overall_pct"],
            "num_participants": best["num_participants"],
            "selection_priority": "max overall accuracy",
        })

    recommendations_path = validation_root / "recommended_head_dilations.csv"
    save_rows(recommendation_rows, recommendations_path)

    print("\nDone.")


if __name__ == "__main__":
    main()