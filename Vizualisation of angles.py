import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the reduced features CSV
df = pd.read_csv('anatomical_motion_features.csv')

# Extract positions
shoulder = df[['shoulder_x', 'shoulder_y', 'shoulder_z']].values
upperarm = df[['upper_arm_x', 'upper_arm_y', 'upper_arm_z']].values
forearm = df[['forearm_x', 'forearm_y', 'forearm_z']].values
hand = df[['hand_x', 'hand_y', 'hand_z']].values

# Plot the joint paths
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(shoulder[:,0], shoulder[:,1], shoulder[:,2], label='Shoulder', c='gray')
ax.plot(upperarm[:,0], upperarm[:,1], upperarm[:,2], label='Upper Arm', c='red')
ax.plot(forearm[:,0], forearm[:,1], forearm[:,2], label='Forearm', c='blue')
ax.plot(hand[:,0], hand[:,1], hand[:,2], label='Hand', c='green')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Arm Joint Centroid Trajectories')
ax.legend()
plt.show()


from scipy.spatial.transform import Rotation as R

# Every Nth frame, plot orientation arrows
step = 25
for idx in range(0, len(df), step):
    pos = upperarm[idx]
    angles = df.loc[idx, ['upper_arm_roll', 'upper_arm_pitch', 'upper_arm_yaw']]
    rot = R.from_euler('xyz', angles.values)
    y_local = rot.apply([0, 1, 0])  # Local 'up' axis

    # Arrow for orientation (Y axis for upper arm)
    ax.quiver(pos[0], pos[1], pos[2],
              y_local[0], y_local[1], y_local[2],
              length=30, color='magenta')

plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['upper_arm_roll'], label='Roll')
plt.plot(df['upper_arm_pitch'], label='Pitch')
plt.plot(df['upper_arm_yaw'], label='Yaw')
plt.legend()
plt.title('Upper Arm Euler Angles Over Time')
plt.xlabel('Frame')
plt.ylabel('Angle (radians)')
plt.show()
