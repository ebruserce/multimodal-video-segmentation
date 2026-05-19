import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================
# CONFIGURE THESE
# ============================================================
ROOT = Path(__file__).parent.parent.parent

GAZE_LOCATIONS_DIR = ROOT / "data/output/gaze_locations_with_metadata"
OUTPUT_DIR         = ROOT / "data/output/group_differences"

# AOI regions to include — None means skip/omit
AOI_REGIONS = ["am_a1head", "am_a2head", "am_activity", "am_bg", "am_screen"]
OMIT_REGIONS = {"am_a1body", "am_a2body", "invalid"}

# Display labels and colors for each region
AOI_DISPLAY = {
    "am_a1head":   ("Head (A1)",   "#E63946"),
    "am_a2head":   ("Head (A2)",   "#F4A261"),
    "am_activity": ("Activity",    "#2A9D8F"),
    "am_bg":       ("Background",  "#457B9D"),
    "am_screen":   ("Screen",      "#9B89B4"),
}

# If True, merge am_a1head and am_a2head into a single "Head" line
MERGE_HEADS = True

SMOOTHING_WINDOW = 30
VIDEO_NAMES = [
    "AM_A1_S5_B2_GA_D1_F1",
    "AM_A3_S0_B0_GA_D1_F0",
    "AM_A5_S2_B5_GM_D1_F1",
    "AM_A7_S4_B6_GM_D1_F1",
]
# ============================================================


def load_gaze_data(video_name):
    folder     = GAZE_LOCATIONS_DIR / video_name
    known_path = folder / f"{video_name}_gaze_locations_with_metadata.csv"
    if not known_path.exists():
        print(f"  No known data found for {video_name}, skipping.")
        return None
    return pd.read_csv(known_path)


def resolve_columns(df, control_variable):
    age_values       = df["age_group"].dropna().unique().tolist()
    diagnosis_values = df["diagnosis_group"].dropna().unique().tolist()
    if control_variable in age_values:
        return "age_group", "diagnosis_group"
    elif control_variable in diagnosis_values:
        return "diagnosis_group", "age_group"
    return None, None


def smooth(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def build_proportion_timecourse(group_df, total_frames, regions):
    """
    For each participant in the group, compute per-frame AOI proportions.
    Then average across participants to get group-level mean and std.

    Returns:
      mean_props: dict of {region: pd.Series} — mean proportion per frame
      std_props:  dict of {region: pd.Series} — std proportion per frame
    """
    participant_ids = group_df["participant_id"].unique()

    # For each participant: build a (total_frames x n_regions) proportion matrix
    all_participant_props = {region: [] for region in regions}

    for pid in participant_ids:
        pid_df = group_df[group_df["participant_id"] == pid].copy()

        # Remove omitted regions
        pid_df = pid_df[~pid_df["predicted_aoi"].isin(OMIT_REGIONS)]

        # Per frame, count how many gaze points fall in each region
        frame_counts = (
            pid_df.groupby(["frame", "predicted_aoi"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=range(total_frames), fill_value=0)
        )

        # Add any missing region columns
        for region in regions:
            if region not in frame_counts.columns:
                frame_counts[region] = 0

        # Convert counts to proportions (row-wise), handle frames with no data
        row_totals = frame_counts[regions].sum(axis=1)
        props = frame_counts[regions].div(row_totals.replace(0, np.nan), axis=0)

        for region in regions:
            all_participant_props[region].append(props[region])

    # Stack across participants and compute mean/std per frame
    mean_props = {}
    std_props  = {}
    for region in regions:
        stacked = pd.concat(all_participant_props[region], axis=1)
        mean_props[region] = stacked.mean(axis=1)
        std_props[region]  = stacked.std(axis=1)

    return mean_props, std_props


def plot_proportion_timecourse(mean_props, std_props, regions,
                                group_label, video_name,
                                output_path, smoothing_window):
    """
    Plot one graph with one line per AOI region showing proportion of gaze over time.
    Y-axis: proportion (0 to 1), sums to 1 across all lines at each frame.
    X-axis: frame number.
    Shaded band: ±1 std across participants.
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    for region in regions:
        label, color = AOI_DISPLAY.get(region, (region, "gray"))
        mean_s = smooth(mean_props[region], smoothing_window)
        std_s  = smooth(std_props[region],  smoothing_window)

        frames = mean_s.index
        ax.plot(frames, mean_s, color=color, linewidth=2, label=label)
        # Comment out to remove std shading
        # ax.fill_between(frames,
        #                 (mean_s - std_s).clip(0, 1),
        #                 (mean_s + std_s).clip(0, 1),
        #                 color=color, alpha=0.15)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Proportion of Gaze")
    ax.set_title(f"{video_name} — {group_label}")
    ax.legend(loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def run(control_variable, group_label_1, group_label_2):
    """
    Generate proportion timecourse plots for all videos and both groups.

    Parameters:
    - control_variable: e.g. "infant", "toddler", "ASD", "NASD"
    - group_label_1:    e.g. "ASD" or "infant"
    - group_label_2:    e.g. "NASD" or "toddler"

    Valid examples:
        run("infant",  "ASD",    "NASD")
        run("toddler", "ASD",    "NASD")
        run("ASD",     "infant", "toddler")
        run("NASD",    "infant", "toddler")

    Outputs 2 graphs per video (one per group), 8 total.
    Each graph has one line per AOI region.
    """
    split_name = f"{control_variable}_{group_label_1}_vs_{group_label_2}"
    split_dir  = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Determine which regions to plot
    if MERGE_HEADS:
        regions = ["am_activity", "am_bg", "am_screen"]
        head_display = ("Head", "#E63946")
        AOI_DISPLAY["head_merged"] = head_display
        plot_regions = ["head_merged"] + regions
    else:
        plot_regions = [r for r in AOI_REGIONS if r in AOI_DISPLAY]

    for video_name in VIDEO_NAMES:
        print(f"\nProcessing: {video_name}")

        df = load_gaze_data(video_name)
        if df is None:
            continue

        # Merge head regions if needed
        if MERGE_HEADS:
            df = df.copy()
            df["predicted_aoi"] = df["predicted_aoi"].replace(
                {"am_a1head": "head_merged", "am_a2head": "head_merged"}
            )
            compute_regions = ["head_merged", "am_activity", "am_bg", "am_screen"]
        else:
            compute_regions = [r for r in AOI_REGIONS if r in AOI_DISPLAY]

        control_col, group_col = resolve_columns(df, control_variable)
        if control_col is None:
            print(f"  '{control_variable}' not found in data, skipping.")
            continue

        df_filtered = df[df[control_col] == control_variable]
        if df_filtered.empty:
            print(f"  No data for '{control_variable}', skipping.")
            continue

        total_frames = int(df_filtered["frame"].max()) + 1

        for group_label in [group_label_1, group_label_2]:
            group_df = df_filtered[df_filtered[group_col] == group_label]

            if group_df.empty:
                print(f"  No data for '{group_label}', skipping.")
                continue

            n_participants = group_df["participant_id"].nunique()
            print(f"  {group_label}: {n_participants} participant(s)")

            mean_props, std_props = build_proportion_timecourse(
                group_df, total_frames, compute_regions
            )

            output_path = split_dir / f"{video_name}_{group_label}_proportions.png"
            plot_proportion_timecourse(
                mean_props, std_props,
                regions=compute_regions,
                group_label=f"{group_label} {control_variable}s (n={n_participants})",
                video_name=video_name,
                output_path=output_path,
                smoothing_window=SMOOTHING_WINDOW,
            )

    print(f"\nDone. Saved to: {split_dir}")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        control = sys.argv[1]
        group1  = sys.argv[2]
        group2  = sys.argv[3]
    else:
        control = "infant"
        group1  = "ASD"
        group2  = "NASD"
        print(f"No args provided, using defaults: {control}, {group1}, {group2}")

    run(control, group1, group2)