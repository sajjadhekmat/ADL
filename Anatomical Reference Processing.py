import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

def extract_markers_for_segment(df, marker_names):
    positions = []
    for marker in marker_names:
        xyz = df[[f'{marker}_x', f'{marker}_y', f'{marker}_z']].values
        positions.append(xyz)
    return np.stack(positions, axis=1)  # (n_frames, n_markers, 3)

def compute_anatomical_orientation(reference_pos, segment_markers, distal_indices):
    """
    reference_pos: (3,) array (e.g. shoulder for upper arm)
    segment_markers: (n_markers, 3) marker positions for 1 frame
    distal_indices: indices of distal markers (closer to elbow or wrist)
    """
    centroid = segment_markers.mean(axis=0)
    # Use specified distal markers to find the opposite segment end
    distal_pos = segment_markers[distal_indices].mean(axis=0)
    y_axis = distal_pos - reference_pos
    y_axis = y_axis / np.linalg.norm(y_axis)
    # For width, pick two markers on opposite sides (use the first and one in the middle)
    lateral_vec = segment_markers[1] - segment_markers[len(segment_markers)//2]
    # Gram-Schmidt orthogonalization
    x_axis = lateral_vec - np.dot(lateral_vec, y_axis) * y_axis
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    rot = R.from_matrix(rotation_matrix)
    euler_angles = rot.as_euler('xyz', degrees=False)
    return centroid, euler_angles

# -- MAIN CODE --

df = pd.read_csv('/Users/sajjadhekmat/PycharmProjects/ADL/CleanedData/1/cleaned_sit_cup_01.csv')

# Marker name lists (based on your columns)
shoulder_marker = 'full_body_trial_4shoulderr'
upper_arm_markers = [f'full_body_trial_4humr{i}' for i in range(1, 9)]
forearm_markers = [f'full_body_trial_4forearmr{i}' for i in range(1, 9)]
hand_markers = [f'full_body_trial_4handr{i}' for i in range(1, 6)]

# Extract xyz for all
n_frames = len(df)
shoulder_positions = df[[f'{shoulder_marker}_x', f'{shoulder_marker}_y', f'{shoulder_marker}_z']].values
upper_arm_positions = extract_markers_for_segment(df, upper_arm_markers)  # (n_frames, 8, 3)
forearm_positions = extract_markers_for_segment(df, forearm_markers)      # (n_frames, 8, 3)
hand_positions = extract_markers_for_segment(df, hand_markers)            # (n_frames, 5, 3)

# Pick distal indices (end markers) for each segment (adjust if needed for your marker set)
upper_arm_distal = [3,4,5,6]      # HumR4-7: Elbow end
forearm_distal = [3,4,5,6]        # Forearm4-7: Wrist end
hand_distal = [3,4]               # HandR4-5: Fingertip end

all_features = []
for i in range(n_frames):
    shoulder_pos = shoulder_positions[i]
    upper_arm_frame = upper_arm_positions[i]
    forearm_frame = forearm_positions[i]
    hand_frame = hand_positions[i]

    # Upper arm orientation (reference: shoulder, segment: upper arm, distal: elbow)
    upper_arm_centroid, upper_arm_euler = compute_anatomical_orientation(
        shoulder_pos, upper_arm_frame, upper_arm_distal
    )
    # Forearm orientation (reference: upper arm centroid ≈ elbow, segment: forearm, distal: wrist)
    forearm_centroid, forearm_euler = compute_anatomical_orientation(
        upper_arm_centroid, forearm_frame, forearm_distal
    )
    # Hand orientation (reference: forearm centroid ≈ wrist, segment: hand, distal: fingertips)
    hand_centroid, hand_euler = compute_anatomical_orientation(
        forearm_centroid, hand_frame, hand_distal
    )

    frame_features = np.concatenate([
        shoulder_pos,               # 3
        upper_arm_centroid,         # 3
        upper_arm_euler,            # 3
        forearm_centroid,           # 3
        forearm_euler,              # 3
        hand_centroid,              # 3
        hand_euler                  # 3
    ])
    all_features.append(frame_features)

all_features = np.array(all_features)   # (n_frames, 21)

# Output column names
col_names = [
    'shoulder_x', 'shoulder_y', 'shoulder_z',
    'upper_arm_x', 'upper_arm_y', 'upper_arm_z',
    'upper_arm_roll', 'upper_arm_pitch', 'upper_arm_yaw',
    'forearm_x', 'forearm_y', 'forearm_z',
    'forearm_roll', 'forearm_pitch', 'forearm_yaw',
    'hand_x', 'hand_y', 'hand_z',
    'hand_roll', 'hand_pitch', 'hand_yaw'
]
pd.DataFrame(all_features, columns=col_names).to_csv('anatomical_motion_features.csv', index=False)
print(f"✅ Done! Saved {all_features.shape[0]} frames × {all_features.shape[1]} features to 'anatomical_motion_features.csv'")
