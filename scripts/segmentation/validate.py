import csv
import yaml
import pandas as pd
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

# Mapping rules from spec:
# am_distractors -> am_bg
# offscreen/invalid -> invalid
AOI_NORMALIZE = {
    "am_distractors": "am_bg",
    "offscreen":      "invalid",
    "invalid":        "invalid",
}

def normalize_aoi(aoi):
    """Normalize gaze data AOI labels to match our predicted labels."""
    return AOI_NORMALIZE.get(aoi, aoi)

def compute_validation(gaze_locations_path, gaze_csv_path):
    """
    Compare predicted AOIs from Script 3 against ground truth AOI column.
    Returns a summary dict and per-region breakdown.
    """
    pred_df  = pd.read_csv(gaze_locations_path)
    truth_df = pd.read_csv(gaze_csv_path)

    # Merge on t
    merged = pd.merge(pred_df, truth_df[["t", "aoi"]], on="t", how="inner")
    merged["aoi_normalized"] = merged["aoi"].apply(normalize_aoi)

    total = len(merged)
    if total == 0:
        print("No matching rows found.")
        return None, None

    merged["match"] = merged["predicted_aoi"] == merged["aoi_normalized"]
    overall_match = merged["match"].sum()
    overall_pct = round(100 * overall_match / total, 2)

    print(f"\nOverall match: {overall_match}/{total} ({overall_pct}%)")

    # Per-region breakdown
    region_stats = []
    for region in merged["aoi_normalized"].unique():
        region_rows = merged[merged["aoi_normalized"] == region]
        region_total = len(region_rows)
        region_match = region_rows["match"].sum()
        region_pct   = round(100 * region_match / region_total, 2)
        region_stats.append({
            "region":       region,
            "total":        region_total,
            "matched":      region_match,
            "match_pct":    region_pct,
            "error_pct":    round(100 - region_pct, 2),
        })
        print(f"  {region}: {region_match}/{region_total} ({region_pct}% match)")

    region_stats.sort(key=lambda x: x["error_pct"], reverse=True)

    summary = {
        "total_rows":    total,
        "matched_rows":  int(overall_match),
        "overall_pct":   overall_pct,
    }

    return summary, region_stats

def save_validation(summary, region_stats, output_path):
    """Save validation results to CSV."""
    rows = [{"type": "overall", "region": "all", **summary,
         "matched": summary["matched_rows"], "total": summary["total_rows"],
         "match_pct": summary["overall_pct"], "error_pct": round(100 - summary["overall_pct"], 2)}]
    for r in region_stats:
        rows.append({"type": "region", **r})

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type", "region", "total", "matched", "match_pct", "error_pct",
            "total_rows", "matched_rows", "overall_pct"
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved validation results to: {output_path}")


if __name__ == "__main__":
    import re
    cfg = load_config()
    ROOT = Path(cfg["paths"].get("project_root", "."))
    video_folder     = ROOT / cfg["paths"]["data"]["input_videos"]
    gaze_folder      = ROOT / cfg["paths"]["data"]["gaze_folder"]
    output_folder    = ROOT / cfg["paths"]["data"]["output"]

    video_files = list(video_folder.glob(cfg["settings"]["video_input_extension"]))
    gaze_files  = list(gaze_folder.glob("*.csv"))

    if not video_files or not gaze_files:
        print("Missing video or gaze files.")
        exit(1)

    video_path = video_files[0]

    def extract_id(fname):
        match = re.search(r'~~([^~]+)~~', fname)
        return match.group(1) if match else fname

    for gaze_path in gaze_files:
        pid = extract_id(gaze_path.name)
        print(f"\nValidating: {pid}")
        gaze_locations_path = output_folder / f"{video_path.stem}_{pid}_gaze_locations.csv"
        if not gaze_locations_path.exists():
            print(f"  Gaze locations file not found, skipping.")
            continue
        summary, region_stats = compute_validation(gaze_locations_path, gaze_path)
        if summary:
            val_out = output_folder / f"{video_path.stem}_{pid}_validation.csv"
            save_validation(summary, region_stats, val_out)

    print("\nDone.")