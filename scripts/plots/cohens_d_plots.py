import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path

# ============================================================
# CONFIGURE THESE
# ============================================================
ROOT = Path(__file__).parent.parent.parent

GAZE_LOCATIONS_DIR = ROOT / "data/output/gaze_locations_with_metadata"
VIDEO_DIR          = ROOT / "data/input/videos"
OUTPUT_DIR         = ROOT / "data/output/cohens_d"

SMOOTHING_WINDOW = 30
TOP_N_FRAMES     = 5      # number of peak difference frames to highlight
MIN_PARTICIPANTS = 3      # skip group if fewer than this many participants

# AOI regions to produce separate plots for
PLOT_REGIONS = ["head_merged", "am_activity", "am_bg"]

# Regions to merge into head_merged
HEAD_REGIONS = {"am_a1head", "am_a2head"}

# Regions to omit entirely
OMIT_REGIONS = {"am_a1body", "am_a2body", "invalid"}

VIDEO_NAMES = [
    "AM_A1_S5_B2_GA_D1_F1",
    "AM_A3_S0_B0_GA_D1_F0",
    "AM_A5_S2_B5_GM_D1_F1",
    "AM_A7_S4_B6_GM_D1_F1",
]

# All four group splits: (control, group1, group2)
GROUP_SPLITS = [
    ("infant",  "ASD",     "NASD"),
    ("toddler", "ASD",     "NASD"),
    ("ASD",     "infant",  "toddler"),
    ("NASD",    "infant",  "toddler"),
]

REGION_LABELS = {
    "head_merged": "Head",
    "am_activity": "Activity",
    "am_bg":       "Background",
}

# ============================================================


def load_gaze_data(video_name):
    folder = GAZE_LOCATIONS_DIR / video_name
    path   = folder / f"{video_name}_gaze_locations_with_metadata.csv"
    if not path.exists():
        print(f"  No data found for {video_name}, skipping.")
        return None
    return pd.read_csv(path)


def resolve_columns(df, control_variable):
    if control_variable in df["age_group"].dropna().unique():
        return "age_group", "diagnosis_group"
    elif control_variable in df["diagnosis_group"].dropna().unique():
        return "diagnosis_group", "age_group"
    return None, None


def smooth(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def build_participant_proportions(group_df, total_frames, regions):
    """
    For each participant, compute proportion of gaze in each region per frame.
    Returns a dict of {region: list of pd.Series (one per participant)}.
    """
    participant_ids = group_df["participant_id"].unique()
    props = {region: [] for region in regions}

    for pid in participant_ids:
        pid_df = group_df[group_df["participant_id"] == pid].copy()
        pid_df = pid_df[~pid_df["predicted_aoi"].isin(OMIT_REGIONS)]

        frame_counts = (
            pid_df.groupby(["frame", "predicted_aoi"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(total_frames), fill_value=0)
        )

        for region in regions:
            if region not in frame_counts.columns:
                frame_counts[region] = 0

        row_totals = frame_counts[regions].sum(axis=1)
        prop_df    = frame_counts[regions].div(row_totals.replace(0, np.nan), axis=0)

        for region in regions:
            props[region].append(prop_df[region])

    return props


def compute_cohens_d(props1, props2, total_frames):
    """
    Compute Cohen's d at each frame between two groups.
    props1, props2: lists of pd.Series (one per participant) for one region.
    Returns a pd.Series of d values indexed by frame.
    """
    arr1 = pd.concat(props1, axis=1).values   # (total_frames, n1)
    arr2 = pd.concat(props2, axis=1).values   # (total_frames, n2)

    mean1 = np.nanmean(arr1, axis=1)
    mean2 = np.nanmean(arr2, axis=1)
    std1  = np.nanstd(arr1,  axis=1, ddof=1)
    std2  = np.nanstd(arr2,  axis=1, ddof=1)
    n1    = arr1.shape[1]
    n2    = arr2.shape[1]

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
    )

    # Avoid division by zero
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.where(pooled_std > 0, (mean1 - mean2) / pooled_std, 0.0)

    return pd.Series(d, index=range(total_frames))


def get_video_frame(video_path, frame_index):
    """Extract a single frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def plot_cohens_d(d_series, region_label, group_label_1, group_label_2,
                  video_name, video_path, top_frames, output_path):
    """
    Plot Cohen's d over frames with reference lines and video frame thumbnails.
    """
    n_thumbs = len(top_frames)

    # Layout: main plot on top, thumbnails on bottom
    fig = plt.figure(figsize=(16, 7 if n_thumbs > 0 else 5))

    if n_thumbs > 0:
        gs = gridspec.GridSpec(2, n_thumbs,
                               height_ratios=[3, 1],
                               hspace=0.4, wspace=0.1)
        ax_main = fig.add_subplot(gs[0, :])
    else:
        ax_main = fig.add_subplot(111)

    frames = d_series.index
    smoothed_d = smooth(d_series, SMOOTHING_WINDOW)

    # Main Cohen's d plot
    ax_main.plot(frames, smoothed_d, color="#333333", linewidth=2)
    ax_main.axhline(0,    color="black",  linewidth=1.0, linestyle="-")
    ax_main.axhline( 0.5, color="#E63946", linewidth=1.0, linestyle="--", alpha=0.7, label="d = ±0.5")
    ax_main.axhline(-0.5, color="#E63946", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_main.axhline( 1.0, color="#C1121F", linewidth=1.0, linestyle="--", alpha=0.7, label="d = ±1.0")
    ax_main.axhline(-1.0, color="#C1121F", linewidth=1.0, linestyle="--", alpha=0.7)

    # Shade above/below zero
    ax_main.fill_between(frames, smoothed_d, 0,
                          where=(smoothed_d >= 0),
                          color="#E63946", alpha=0.15,
                          label=f"{group_label_1} higher")
    ax_main.fill_between(frames, smoothed_d, 0,
                          where=(smoothed_d < 0),
                          color="#457B9D", alpha=0.15,
                          label=f"{group_label_2} higher")

    # Mark top frames on the plot
    for frame_idx in top_frames:
        ax_main.axvline(frame_idx, color="gray", linewidth=0.8,
                        linestyle=":", alpha=0.8)

    ax_main.set_xlabel("Frame")
    ax_main.set_ylabel("Cohen's d")
    ax_main.set_title(
        f"{video_name} — {region_label}\n"
        f"{group_label_1} vs {group_label_2}  "
        f"(positive = {group_label_1} higher)"
    )
    ax_main.legend(loc="upper right", fontsize=8)
    ax_main.grid(axis='y', linestyle='--', alpha=0.3)

    # Thumbnail row
    if n_thumbs > 0 and video_path.exists():
        for i, frame_idx in enumerate(top_frames):
            ax_thumb = fig.add_subplot(gs[1, i])
            img = get_video_frame(video_path, frame_idx)
            if img is not None:
                ax_thumb.imshow(img)
            ax_thumb.set_title(f"f={frame_idx}", fontsize=7)
            ax_thumb.axis("off")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def run_split(control_variable, group_label_1, group_label_2):
    split_name = f"{control_variable}_{group_label_1}_vs_{group_label_2}"
    split_dir  = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Split: {split_name}")
    print(f"{'='*60}")

    for video_name in VIDEO_NAMES:
        print(f"\n  Video: {video_name}")

        df = load_gaze_data(video_name)
        if df is None:
            continue

        # Merge head regions
        df = df.copy()
        df["predicted_aoi"] = df["predicted_aoi"].replace(
            {r: "head_merged" for r in HEAD_REGIONS}
        )

        control_col, group_col = resolve_columns(df, control_variable)
        if control_col is None:
            print(f"    '{control_variable}' not found in data, skipping.")
            continue

        df_filtered = df[df[control_col] == control_variable]
        if df_filtered.empty:
            print(f"    No data for '{control_variable}', skipping.")
            continue

        total_frames = int(df_filtered["frame"].max()) + 1

        # Get data for each group
        g1_df = df_filtered[df_filtered[group_col] == group_label_1]
        g2_df = df_filtered[df_filtered[group_col] == group_label_2]

        n1 = g1_df["participant_id"].nunique()
        n2 = g2_df["participant_id"].nunique()
        print(f"    {group_label_1}: {n1} participants, "
              f"{group_label_2}: {n2} participants")

        if n1 < MIN_PARTICIPANTS or n2 < MIN_PARTICIPANTS:
            print(f"    Not enough participants (min={MIN_PARTICIPANTS}), skipping.")
            continue

        # Build per-participant proportions for both groups
        props1 = build_participant_proportions(g1_df, total_frames, PLOT_REGIONS)
        props2 = build_participant_proportions(g2_df, total_frames, PLOT_REGIONS)

        video_path = VIDEO_DIR / f"{video_name}.avi"

        # One plot per AOI region
        for region in PLOT_REGIONS:
            if not props1[region] or not props2[region]:
                continue

            d_series = compute_cohens_d(props1[region], props2[region], total_frames)
            smoothed_d = smooth(d_series, SMOOTHING_WINDOW)

            # Find top N frames by absolute d value
            top_frames = (
                smoothed_d.abs()
                .nlargest(TOP_N_FRAMES)
                .index
                .tolist()
            )
            top_frames = sorted(top_frames)

            region_label = REGION_LABELS.get(region, region)
            output_path  = split_dir / f"{video_name}_{region}_cohens_d.png"

            plot_cohens_d(
                d_series, region_label,
                group_label_1, group_label_2,
                video_name, video_path,
                top_frames, output_path
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for control, group1, group2 in GROUP_SPLITS:
        run_split(control, group1, group2)
    print("\nAll splits done.")


if __name__ == "__main__":
    main()