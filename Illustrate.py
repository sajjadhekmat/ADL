import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# 1. Load the cleaned CSV (use your actual local path)
df = pd.read_csv('/Users/sajjadhekmat/PycharmProjects/ADL/CleanedData/2/cleaned_mug_prox_03.csv')

# 2. Define your markers
shoulder = 'shoulderr'
elbow_markers = [
    'humr3',
    'humr4',
    'humr5',
    'humr6'
]
hand_markers = [
    'handr1',
    'handr2',
    'handr3',
    'handr4',
    'handr5'
]

# 3. Compute averaged marker positions
def avg_marker_pos(df, markers):
    coords = np.stack([df[[f"{m}_x", f"{m}_y", f"{m}_z"]].values
                       for m in markers if f"{m}_x" in df.columns], axis=2)
    return coords.mean(axis=2)

elbow_xyz = avg_marker_pos(df, elbow_markers)
hand_xyz  = avg_marker_pos(df, hand_markers)
shoulder_xyz = df[[f"{shoulder}_x", f"{shoulder}_y", f"{shoulder}_z"]].values

# 4. Plot 3D trajectories
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(121, projection='3d')
ax.plot(shoulder_xyz[:,0], shoulder_xyz[:,1], shoulder_xyz[:,2], label='Shoulder')
ax.plot(elbow_xyz[:,0],    elbow_xyz[:,1],    elbow_xyz[:,2],    label='Elbow')
ax.plot(hand_xyz[:,0],     hand_xyz[:,1],     hand_xyz[:,2],     label='Hand')
ax.set_title('3D Trajectories')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()

# 5. Compute and plot hand velocity
velocity = np.linalg.norm(np.diff(hand_xyz, axis=0), axis=1)
ax2 = fig.add_subplot(122)
ax2.plot(velocity)
ax2.set_title('Hand Velocity (mm/frame)')
ax2.set_xlabel('Frame'); ax2.set_ylabel('Velocity')

# Make sure the 'plot' folder exists
if not os.path.exists('plot'):
    os.makedirs('plot')

# Save the figure before showing it
plt.savefig('plot/trajectory_and_velocity.png', dpi=300)  # you can change the filename and format

plt.tight_layout()
plt.show()
