import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
from glob import glob

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_ROOT = "/Users/sajjadhekmat/PycharmProjects/ADL/CleanedData"  # root folder with subfolders 1-12
OUTPUT_ROOT = "/Users/sajjadhekmat/PycharmProjects/ADL/SegmentedData"  # mirrored output root

SMOOTH_SIGMA_POS = 8
SMOOTH_SIGMA_SPEED = 8

# ==========================================
# FUNCTION TO PROCESS SINGLE CSV
# ==========================================
def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    # Extract hand centroid
    hand_cols_x = [c for c in df.columns if "hand" in c and c.endswith("_x")]
    hand_cols_y = [c for c in df.columns if "hand" in c and c.endswith("_y")]
    hand_cols_z = [c for c in df.columns if "hand" in c and c.endswith("_z")]

    hand_pos = np.column_stack([
        df[hand_cols_x].mean(axis=1),
        df[hand_cols_y].mean(axis=1),
        df[hand_cols_z].mean(axis=1)
    ])

    # Smooth position
    hand_pos_s = gaussian_filter1d(hand_pos, sigma=SMOOTH_SIGMA_POS, axis=0)

    # Compute 3D speed
    speed = np.linalg.norm(np.diff(hand_pos_s, axis=0), axis=1)
    speed = np.insert(speed, 0, 0)
    speed_s = gaussian_filter1d(speed, sigma=SMOOTH_SIGMA_SPEED)

    # Adaptive threshold
    baseline = np.percentile(speed_s, 30)
    std = np.std(speed_s)
    threshold = baseline + 0.30 * std

    active_idx = np.where(speed_s > threshold)[0]

    if len(active_idx) == 0:
        start_frame, end_frame = 0, len(df) - 1
    else:
        splits = np.where(np.diff(active_idx) > 1)[0] + 1
        runs = np.split(active_idx, splits)
        reach_run = runs[0]
        start_frame = 0
        end_frame = int(reach_run[-1])

    # Save segmented CSV
    seg_df = df.iloc[start_frame:end_frame+1]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    seg_df.to_csv(output_file, index=False)
    print(f"Saved CSV: {output_file} | Frames: {len(seg_df)}")

    # Save plot with same base name as CSV
    plt.figure(figsize=(12, 5))
    plt.plot(speed_s, label="3D Hand Speed", color="blue")
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
    plt.axvspan(start_frame, end_frame, color="green", alpha=0.3, label="Reach Segment")
    plt.title(f"Segmentation: {os.path.basename(input_file)}")
    plt.xlabel("Frame")
    plt.ylabel("Speed (mm/frame)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.splitext(output_file)[0] + ".png"
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved plot: {plot_file}")


# ==========================================
# PROCESS ALL CSV FILES IN ALL SUBFOLDERS
# ==========================================
for subfolder in range(1, 13):
    input_folder = os.path.join(INPUT_ROOT, str(subfolder))
    csv_files = glob(os.path.join(input_folder, "*.csv"))

    for input_file in csv_files:
        # mirrored output path
        rel_path = os.path.relpath(input_file, INPUT_ROOT)
        output_file = os.path.join(OUTPUT_ROOT, rel_path)

        process_csv(input_file, output_file)
