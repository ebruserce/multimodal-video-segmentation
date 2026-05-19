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

# AOI numerical values — None means skip/omit
AOI_VALUES = {
    "am_a1head":   5,
    "am_a2head":   5,
    "am_activity": 4,
    "am_bg":       3,
    "am_screen":   2,
    "am_a1body":   None,
    "am_a2body":   None,
    "invalid":     None,
}

AOI_LABELS = {
    5: "Head",
    4: "Activity",
    3: "Background",
    2: "Screen border",
}

SMOOTHING_WINDOW = 30
VIDEO_NAMES = [
    "AM_A1_S5_B2_GA_D1_F1",
    "AM_A3_S0_B0_GA_D1_F0",
    "AM_A5_S2_B5_GM_D1_F1",
    "AM_A7_S4_B6_GM_D1_F1",
]
# ============================================================


def load_gaze_data(video_name):
    """Load gaze locations with metadata for a given video."""
    folder = GAZE_LOCATIONS_DIR / video_name
    known_path = folder / f"{video_name}_gaze_locations_with_metadata.csv"
    if not known_path.exists():
        print(f"  No known data found for {video_name}, skipping.")
        return None
    df = pd.read_csv(known_path)
    return df


def get_aoi_value(aoi):
    """Convert AOI label to numerical value. Returns None if omitted."""
    return AOI_VALUES.get(aoi, None)


def build_participant_timecourse(participant_df, total_frames):
    """
    For a single participant, build a per-frame AOI value series.
    For frames with multiple gaze points, take the most common valid AOI.
    Returns a pd.Series indexed by frame, with NaN for frames with no valid data.
    """
    participant_df = participant_df.copy()
    participant_df["aoi_value"] = participant_df["predicted_aoi"].apply(get_aoi_value)
    participant_df = participant_df.dropna(subset=["aoi_value"])

    if participant_df.empty:
        return pd.Series(np.nan, index=range(total_frames))

    frame_aoi = (
        participant_df.groupby("frame")["aoi_value"]
        .agg(lambda x: x.mode()[0])
    )

    return frame_aoi.reindex(range(total_frames))


def build_group_timecourse(group_df, total_frames):
    """
    For a group of participants, compute mean and std AOI value per frame.
    Returns mean and std Series indexed by frame.
    """
    participant_ids = group_df["participant_id"].unique()
    timecourses = []

    for pid in participant_ids:
        pid_df = group_df[group_df["participant_id"] == pid]
        tc = build_participant_timecourse(pid_df, total_frames)
        timecourses.append(tc)

    if not timecourses:
        return None, None

    stacked = pd.concat(timecourses, axis=1)
    mean_tc = stacked.mean(axis=1)
    std_tc  = stacked.std(axis=1)
    return mean_tc, std_tc


def smooth(series, window):
    """Apply rolling mean smoothing."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def plot_group_timecourse(mean_tc, std_tc, group_label, color,
                           video_name, output_path, smoothing_window):
    """Plot a single group's smoothed timecourse and save it."""
    frames = mean_tc.index
    smoothed_mean = smooth(mean_tc, smoothing_window)
    smoothed_std  = smooth(std_tc,  smoothing_window)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(frames, smoothed_mean, color=color, linewidth=2, label=group_label)
    ax.fill_between(frames,
                    smoothed_mean - smoothed_std,
                    smoothed_mean + smoothed_std,
                    color=color, alpha=0.2)

    ax.set_yticks(sorted(AOI_LABELS.keys()))
    ax.set_yticklabels([AOI_LABELS[k] for k in sorted(AOI_LABELS.keys())])
    ax.set_ylim(1.5, 5.5)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Region of Interest")
    ax.set_title(f"{video_name} — {group_label}")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def resolve_columns(df, control_variable):
    """
    Determine which column to filter by for control and which for group split.
    Returns (control_col, group_col) or (None, None) if control_variable not found.
    """
    age_values      = df["age_group"].dropna().unique().tolist()
    diagnosis_values = df["diagnosis_group"].dropna().unique().tolist()

    if control_variable in age_values:
        return "age_group", "diagnosis_group"
    elif control_variable in diagnosis_values:
        return "diagnosis_group", "age_group"
    else:
        return None, None


def run(control_variable, group_label_1, group_label_2):
    """
    Generate group difference plots for all videos.

    Parameters:
    - control_variable: value to filter on, e.g. "infant", "toddler", "ASD", "NASD"
    - group_label_1:    first group to compare, e.g. "ASD" or "infant"
    - group_label_2:    second group to compare, e.g. "NASD" or "toddler"

    Valid call examples:
        run("infant",  "ASD",     "NASD")
        run("toddler", "ASD",     "NASD")
        run("ASD",     "infant",  "toddler")
        run("NASD",    "infant",  "toddler")

    Outputs 2 graphs per video (one per group), 8 total per call.
    """
    split_name = f"{control_variable}_{group_label_1}_vs_{group_label_2}"
    split_dir  = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        group_label_1: "#E63946",   # red
        group_label_2: "#457B9D",   # blue
    }

    for video_name in VIDEO_NAMES:
        print(f"\nProcessing: {video_name}")

        df = load_gaze_data(video_name)
        if df is None:
            continue

        # Determine which columns to use based on control_variable
        control_col, group_col = resolve_columns(df, control_variable)
        if control_col is None:
            print(f"  '{control_variable}' not found in age_group or diagnosis_group columns, skipping.")
            continue

        # Filter to control group
        df_filtered = df[df[control_col] == control_variable]
        if df_filtered.empty:
            print(f"  No data for '{control_variable}', skipping.")
            continue

        # Verify both group labels exist in the group column
        available_groups = df_filtered[group_col].unique().tolist()
        for group_label in [group_label_1, group_label_2]:
            if group_label not in available_groups:
                print(f"  Warning: '{group_label}' not found in {group_col} "
                      f"(available: {available_groups})")

        total_frames = int(df_filtered["frame"].max()) + 1

        for group_label in [group_label_1, group_label_2]:
            group_df = df_filtered[df_filtered[group_col] == group_label]

            if group_df.empty:
                print(f"  No data for group '{group_label}', skipping.")
                continue

            n_participants = group_df["participant_id"].nunique()
            print(f"  {group_label}: {n_participants} participant(s)")

            mean_tc, std_tc = build_group_timecourse(group_df, total_frames)
            if mean_tc is None:
                print(f"  Could not build timecourse for {group_label}, skipping.")
                continue

            output_path = split_dir / f"{video_name}_{group_label}.png"
            plot_group_timecourse(
                mean_tc, std_tc,
                group_label=f"{group_label} {control_variable}s (n={n_participants})",
                color=colors[group_label],
                video_name=video_name,
                output_path=output_path,
                smoothing_window=SMOOTHING_WINDOW,
            )

    print(f"\nDone. Saved to: {split_dir}")


if __name__ == "__main__":
    # Usage: python group_differences.py <control> <group1> <group2>
    # Examples:
    #   python group_differences.py infant ASD NASD
    #   python group_differences.py toddler ASD NASD
    #   python group_differences.py ASD infant toddler
    #   python group_differences.py NASD infant toddler
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