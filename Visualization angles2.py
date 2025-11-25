import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation as R

# Load CSV
df = pd.read_csv('anatomical_motion_features.csv')

# Estimate segment lengths from the first frame
upper_len = np.linalg.norm([
    df.loc[0, 'upper_arm_x'] - df.loc[0, 'shoulder_x'],
    df.loc[0, 'upper_arm_y'] - df.loc[0, 'shoulder_y'],
    df.loc[0, 'upper_arm_z'] - df.loc[0, 'shoulder_z']
])

forearm_len = np.linalg.norm([
    df.loc[0, 'forearm_x'] - df.loc[0, 'upper_arm_x'],
    df.loc[0, 'forearm_y'] - df.loc[0, 'upper_arm_y'],
    df.loc[0, 'forearm_z'] - df.loc[0, 'upper_arm_z']
])

hand_len = np.linalg.norm([
    df.loc[0, 'hand_x'] - df.loc[0, 'forearm_x'],
    df.loc[0, 'hand_y'] - df.loc[0, 'forearm_y'],
    df.loc[0, 'hand_z'] - df.loc[0, 'forearm_z']
])

# 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], 'o-', lw=3, color='red')

# Set axis limits (adjust if needed)
ax.set_xlim(-500, 500)
ax.set_ylim(-800, 0)
ax.set_zlim(0, 500)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Arm Animation Using Euler Angles')
ax.view_init(elev=20., azim=-60)


# Convert roll/pitch/yaw to rotation matrix
def get_rotation_matrix(roll, pitch, yaw):
    return R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()


# Initialize line
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,


# Update for each frame
def update(frame):
    shoulder = np.array([df.loc[frame, 'shoulder_x'], df.loc[frame, 'shoulder_y'], df.loc[frame, 'shoulder_z']])

    upper_rot = get_rotation_matrix(df.loc[frame, 'upper_arm_roll'],
                                    df.loc[frame, 'upper_arm_pitch'],
                                    df.loc[frame, 'upper_arm_yaw'])
    upper_vec = upper_rot @ np.array([upper_len, 0, 0])
    upper_pos = shoulder + upper_vec

    forearm_rot = get_rotation_matrix(df.loc[frame, 'forearm_roll'],
                                      df.loc[frame, 'forearm_pitch'],
                                      df.loc[frame, 'forearm_yaw'])
    forearm_vec = forearm_rot @ np.array([forearm_len, 0, 0])
    forearm_pos = upper_pos + forearm_vec

    hand_rot = get_rotation_matrix(df.loc[frame, 'hand_roll'],
                                   df.loc[frame, 'hand_pitch'],
                                   df.loc[frame, 'hand_yaw'])
    hand_vec = hand_rot @ np.array([hand_len, 0, 0])
    hand_pos = forearm_pos + hand_vec

    # Combine positions for plotting
    x = [shoulder[0], upper_pos[0], forearm_pos[0], hand_pos[0]]
    y = [shoulder[1], upper_pos[1], forearm_pos[1], hand_pos[1]]
    z = [shoulder[2], upper_pos[2], forearm_pos[2], hand_pos[2]]

    line.set_data(x, y)
    line.set_3d_properties(z)
    return line,


# Create animation
ani = FuncAnimation(fig, update, frames=len(df), init_func=init, blit=True, interval=50)

# Save as GIF
ani.save("arm_angle_based.gif", writer=PillowWriter(fps=20))
print("Saved animation as arm_angle_based.gif")
