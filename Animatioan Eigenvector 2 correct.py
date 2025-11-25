import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.spatial.transform import Rotation as R

# -------------------
# Load angles CSV
# -------------------
angles_df = pd.read_csv("/Users/sajjadhekmat/PycharmProjects/ADL/CleanedData/angles_box_b2t_01.csv")
angles_df.columns = angles_df.columns.str.strip()
print("Columns in CSV:", angles_df.columns.tolist())

# -------------------
# Arm geometry (units: cm)
# -------------------
L1 = 18
L2 = 22
L3 = 10


# -------------------
# Forward kinematics
# -------------------
def fk_7dof(shoulder, elbow, forearm, wrist):
    R_sh = R.from_euler('xyz', shoulder, degrees=True).as_matrix()
    R_el = R.from_euler('y', elbow, degrees=True).as_matrix()
    R_fr = R.from_euler('z', forearm, degrees=True).as_matrix()
    R_wr = R.from_euler('xy', wrist, degrees=True).as_matrix()

    shoulder_pos = np.array([0, 0, 0])
    elbow_pos = shoulder_pos + R_sh @ np.array([0, 0, L1])
    wrist_pos = elbow_pos + (R_sh @ R_el @ np.array([0, 0, L2]))
    hand_pos = wrist_pos + (R_sh @ R_el @ R_fr @ R_wr @ np.array([0, 0, L3]))

    return shoulder_pos, elbow_pos, wrist_pos, hand_pos


# -------------------
# Setup figure
# -------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(0, 60)
line, = ax.plot([], [], [], 'o-', lw=3)


# -------------------
# Update function
# -------------------
def update(i):
    row = angles_df.iloc[i]
    try:
        shoulder = [row['shoulder_x'], row['shoulder_y'], row['shoulder_z']]
        elbow = row['elbow']
        forearm = row['forearm']
        wrist = [row['wrist_x'], row['wrist_y']]
    except KeyError:
        shoulder = [row[angles_df.columns[1]], row[angles_df.columns[2]], row[angles_df.columns[3]]]
        elbow = row[angles_df.columns[4]]
        forearm = row[angles_df.columns[5]]
        wrist = [row[angles_df.columns[6]], row[angles_df.columns[7]]]

    p0, p1, p2, p3 = fk_7dof(shoulder, elbow, forearm, wrist)
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    zs = [p0[2], p1[2], p2[2], p3[2]]

    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    ax.set_title(f"Frame {i}")

    return line,


# -------------------
# Create animation
# -------------------
ani = FuncAnimation(fig, update, frames=len(angles_df), interval=30)

# -------------------
# Save as MP4
# -------------------
# Make sure ffmpeg is installed: https://ffmpeg.org/download.html
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save("/Users/sajjadhekmat/PycharmProjects/ADL/arm_animation.mp4", writer=writer)

print("Animation saved as MP4!")
