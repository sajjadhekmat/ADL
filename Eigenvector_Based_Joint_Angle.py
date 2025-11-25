import numpy as np
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R

# ===============================
# USER SETTINGS
# ===============================
INPUT_ROOT = "/Users/sajjadhekmat/PycharmProjects/ADL/CleanedData"      # folder with subfolders 1-12
OUTPUT_ROOT = "/Users/sajjadhekmat/PycharmProjects/ADL/AnglesOutput"    # mirrored folder structure for angles

# ===============================
# 1. HELPER FUNCTIONS
# ===============================

def markers_from_df(frame_row, marker_names):
    pts = []
    for m in marker_names:
        xk, yk, zk = f"{m}_x", f"{m}_y", f"{m}_z"
        if xk in frame_row.index:
            vals = frame_row[[xk, yk, zk]].to_numpy(dtype=float)
            if not np.any(np.isnan(vals)):
                pts.append(vals)
    return np.vstack(pts) if pts else np.empty((0, 3))


def svd_orientation(markers):
    if len(markers) < 3:
        return np.eye(3), np.array([np.nan]*3)

    centroid = markers.mean(axis=0)
    centered = markers - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    Rm = Vt.T

    if np.linalg.det(Rm) < 0:
        Rm[:, -1] *= -1

    return Rm, centroid


def enforce_axis_continuity(R_prev, R_cur):
    if R_prev is None:
        return R_cur

    R_fix = R_cur.copy()
    for i in range(3):
        if np.dot(R_prev[:, i], R_cur[:, i]) < 0:
            R_fix[:, i] *= -1
    return R_fix


def joint_angles(Rp_ref, Rc_ref, Rp_cur, Rc_cur):
    R_ref = Rp_ref.T @ Rc_ref
    R_cur = Rp_cur.T @ Rc_cur
    R_delta = R_ref.T @ R_cur
    return R.from_matrix(R_delta).as_euler("xyz", degrees=False)


# ===============================
# 2. MARKER GROUPS
# ===============================
upper_arm_markers = [f"humr{i}" for i in range(1, 9)]
forearm_markers = [f"forearmr{i}" for i in range(1, 9)]
hand_markers = [f"handr{i}" for i in range(1, 6)]
torso_markers = [
    "c7", "strtop", "strbottom", "t8", "torsoside",
    "pelvisbl", "pelvisbr", "pelvisfl", "pelvisfr"
]

# ===============================
# 3. PROCESS ALL FOLDERS AND FILES
# ===============================

for folder_name in map(str, range(1, 13)):
    input_folder = os.path.join(INPUT_ROOT, folder_name)
    output_folder = os.path.join(OUTPUT_ROOT, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".csv"):
            continue

        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name.replace("cleaned", "angles"))

        print(f"Processing {input_path} → {output_path}")

        # Load CSV
        df = pd.read_csv(input_path)

        # Find neutral frame
        upper_centroids = []
        torso_centroids = []
        for i in range(len(df)):
            row = df.iloc[i]
            upper_centroids.append(markers_from_df(row, upper_arm_markers).mean(axis=0))
            torso_centroids.append(markers_from_df(row, torso_markers).mean(axis=0))

        upper_centroids = np.array(upper_centroids)
        torso_centroids = np.array(torso_centroids)
        dist = np.linalg.norm(upper_centroids - torso_centroids, axis=1)
        neutral_frame = np.nanargmin(dist)

        row0 = df.iloc[neutral_frame]
        R_torso_ref, _ = svd_orientation(markers_from_df(row0, torso_markers))
        R_upper_ref, _ = svd_orientation(markers_from_df(row0, upper_arm_markers))
        R_forearm_ref, _ = svd_orientation(markers_from_df(row0, forearm_markers))
        R_hand_ref, _ = svd_orientation(markers_from_df(row0, hand_markers))

        # Compute angles
        R_torso_prev = R_upper_prev = R_forearm_prev = R_hand_prev = None
        angles = []

        for i in range(len(df)):
            row = df.iloc[i]
            R_torso, _ = svd_orientation(markers_from_df(row, torso_markers))
            R_upper, _ = svd_orientation(markers_from_df(row, upper_arm_markers))
            R_forearm, _ = svd_orientation(markers_from_df(row, forearm_markers))
            R_hand, _ = svd_orientation(markers_from_df(row, hand_markers))

            R_torso = enforce_axis_continuity(R_torso_prev, R_torso)
            R_upper = enforce_axis_continuity(R_upper_prev, R_upper)
            R_forearm = enforce_axis_continuity(R_forearm_prev, R_forearm)
            R_hand = enforce_axis_continuity(R_hand_prev, R_hand)

            R_torso_prev, R_upper_prev = R_torso, R_upper
            R_forearm_prev, R_hand_prev = R_forearm, R_hand

            shoulder = joint_angles(R_torso_ref, R_upper_ref, R_torso, R_upper)
            elbow = joint_angles(R_upper_ref, R_forearm_ref, R_upper, R_forearm)
            wrist = joint_angles(R_forearm_ref, R_hand_ref, R_forearm, R_hand)

            angles.append([
                shoulder[0], shoulder[1], shoulder[2],
                elbow[1],
                elbow[2],
                wrist[0], wrist[1]
            ])

        angles_df = pd.DataFrame(angles, columns=[
            "shoulder_roll", "shoulder_pitch", "shoulder_yaw",
            "elbow_pitch", "forearm_twist",
            "wrist_flexion", "wrist_deviation"
        ])
        angles_df.to_csv(output_path, index=False)
        print(f"Saved angles → {output_path}")

print("All folders processed successfully.")
