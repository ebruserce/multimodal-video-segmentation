# Ebru Serce, 2026
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


def normalize_diagnosis(dx):
    """
    Convert original diagnosis labels into analysis groups.
    ASD stays ASD.
    NT and ATP become NASD.
    """
    if pd.isna(dx):
        return "unknown"

    dx = str(dx).strip()

    if dx == "ASD":
        return "ASD"

    if dx in ["NT", "ATP"]:
        return "NASD"

    return dx


def assign_age_group(age_months):
    """
    Assign age group based on corrected age in months.
    Infant:  < 18 months
    Toddler: >= 18 months
    """
    if pd.isna(age_months):
        return "unknown"

    age_months = float(age_months)

    if age_months < 18:
        return "infant"

    return "toddler"


def load_metadata(metadata_path):
    """
    Load participant metadata and prepare columns needed for analysis.
    Expected columns include:
      - edf
      - dx
      - et_age_corrected_months
    """
    metadata_df = pd.read_csv(metadata_path)

    required_cols = ["edf", "dx", "et_age_corrected_months"]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]

    if missing_cols:
        raise ValueError(f"Metadata file is missing required columns: {missing_cols}")

    metadata_df = metadata_df.copy()

    metadata_df["participant_id"] = metadata_df["edf"].astype(str).str.strip()
    metadata_df["dx"] = metadata_df["dx"].astype(str).str.strip()
    metadata_df["diagnosis_group"] = metadata_df["dx"].apply(normalize_diagnosis)
    metadata_df["age_months"] = pd.to_numeric(
        metadata_df["et_age_corrected_months"],
        errors="coerce"
    )
    metadata_df["age_group"] = metadata_df["age_months"].apply(assign_age_group)
    metadata_df["group_label"] = (
        metadata_df["age_group"] + "_" + metadata_df["diagnosis_group"]
    )

    keep_cols = [
        "participant_id",
        "dx",
        "diagnosis_group",
        "age_months",
        "age_group",
        "group_label",
        "risk_group",
        "visit",
        "id",
    ]

    keep_cols = [col for col in keep_cols if col in metadata_df.columns]

    metadata_df = metadata_df[keep_cols].drop_duplicates(subset=["participant_id"])

    return metadata_df


def attach_metadata_to_file(gaze_locations_path, metadata_df, output_path):
    """
    Attach participant metadata to one per-video gaze-location CSV.

    Saves:
      1. Main file with only rows that have metadata
      2. Separate missing-metadata file for rows without metadata
    """
    gaze_df = pd.read_csv(gaze_locations_path)

    if "participant_id" not in gaze_df.columns:
        raise ValueError(f"{gaze_locations_path} is missing participant_id column.")

    merged_df = gaze_df.merge(
        metadata_df,
        on="participant_id",
        how="left"
    )

    has_metadata = merged_df["diagnosis_group"].notna()

    matched_df = merged_df[has_metadata].copy()
    missing_df = merged_df[~has_metadata].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    matched_df.to_csv(output_path, index=False)

    missing_output_path = output_path.parent / output_path.name.replace(
        "_with_metadata.csv",
        "_missing_metadata.csv"
    )
    missing_df.to_csv(missing_output_path, index=False)

    print(f"Saved metadata-enriched file to: {output_path}")
    print(f"  Kept rows with metadata: {len(matched_df)}")
    print(f"  Saved rows missing metadata: {len(missing_df)}")

    if len(missing_df) > 0:
        missing_participants = sorted(
            missing_df["participant_id"].dropna().unique()
        )

        print(f"  Missing participant IDs: {missing_participants}")
        print(f"  Missing metadata file: {missing_output_path}")

    return matched_df


def attach_metadata_all_videos(gaze_locations_root, metadata_path, output_root):
    """
    Find all per-video gaze-location CSVs and attach participant metadata.
    """
    metadata_df = load_metadata(metadata_path)

    input_files = sorted(
        gaze_locations_root.glob("*/*_gaze_locations_all.csv")
    )

    if not input_files:
        print(f"No gaze-location files found in: {gaze_locations_root}")
        return

    print(f"Found {len(input_files)} gaze-location file(s).")

    for gaze_locations_path in input_files:
        video_name = gaze_locations_path.parent.name

        print(f"\nProcessing video: {video_name}")

        output_folder = output_root / video_name
        output_path = output_folder / f"{video_name}_gaze_locations_with_metadata.csv"

        attach_metadata_to_file(
            gaze_locations_path=gaze_locations_path,
            metadata_df=metadata_df,
            output_path=output_path
        )

    print("\nDone attaching metadata.")


if __name__ == "__main__":
    cfg = load_config()

    ROOT = Path(cfg["paths"].get("project_root", "."))

    input_folder = ROOT / cfg["paths"]["data"]["input"]
    output_folder = ROOT / cfg["paths"]["data"]["output"]

    metadata_path = input_folder / "am_edf_ch_mapping_v01.csv"

    gaze_locations_root = output_folder / "gaze_locations"
    metadata_output_root = output_folder / "gaze_locations_with_metadata"

    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        exit(1)

    attach_metadata_all_videos(
        gaze_locations_root=gaze_locations_root,
        metadata_path=metadata_path,
        output_root=metadata_output_root
    )