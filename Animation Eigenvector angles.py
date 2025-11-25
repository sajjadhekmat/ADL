"""
3D skeleton animation (stick figure + eigenvector arrows) from cleaned marker CSV.

Usage:
- Put your cleaned CSV path in INPUT_CSV.
- Run in a Jupyter notebook (displays inline) or run as a script (writes skeleton_anim.mp4 if ffmpeg available).
- Requires: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- User config ----------
INPUT_CSV = "/Users/sajjadhekmat/PycharmProjects/ADL/CleanedData/1/cleaned_box_b2t_01.csv"
OUTPUT_MOV = "skeleton_anim.mp4"                      # saved if running as script
FRAME_INTERVAL_MS = 40                                # playback speed
ARROW_SCALE = 80                                      # arrow length (tweak to fit units)
SHOW_AXIS = True                                      # show coordinate axes on plot
# ----------------------------------

# Marker groups (match your cleaned columns)
TORSO_MARKERS = ['c7','strtop','strbottom','t8','torsoside','pelvisbl','pelvisbr','pelvisfl','pelvisfr']
UPPER_MARKERS = [f"humr{i}" for i in range(1,9)]
FOREARM_MARKERS = [f"forearmr{i}" for i in range(1,9)]
HAND_MARKERS = [f"handr{i}" for i in range(1,6)]

# ---------- Helpers ----------
def load_df(path):
    df = pd.read_csv(path)
    return df

def group_exists(df, m):
    return all(f"{m}_{axis}" in df.columns for axis in ('x','y','z'))

def collect_group(df, markers):
    """
    Returns array shape (n_frames, n_markers_present, 3).
    If a marker is not present it's skipped.
    """
    arrs = []
    for m in markers:
        if group_exists(df, m):
            arrs.append(df[[f"{m}_x", f"{m}_y", f"{m}_z"]].to_numpy(dtype=float))
    if not arrs:
        return None
    return np.stack(arrs, axis=1)  # frames, markers, xyz

def safe_mean(points):
    """Mean that tolerates NaNs (points: (N,3)). returns (3,)"""
    if points.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    return np.nanmean(points, axis=0)

def svd_orientation(points):
    """
    points: (n_markers, 3)
    returns (R, centroid) where R is 3x3 rotation (columns are eigenvectors),
    centroid is mean position. On failure returns (eye, centroid).
    """
    # safety
    if points is None or points.size == 0:
        return np.eye(3), np.array([np.nan, np.nan, np.nan])
    if np.isnan(points).any():
        # drop Na rows
        ok = ~np.isnan(points).any(axis=1)
        points = points[ok]
    if points.shape[0] < 3:
        # insufficient markers -> return identity orientation but correct centroid
        return np.eye(3), safe_mean(points)
    centroid = points.mean(axis=0)
    centered = points - centroid
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(3), centroid
    R = Vt.T
    # ensure right-handed
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R, centroid

def ensure_continuity(prev_R, cur_R):
    """
    Flip axes of cur_R if needed so they are aligned with prev_R to avoid 180-degree flips.
    This compares each column vector and flips sign if dot < 0.
    """
    if prev_R is None:
        return cur_R
    R = cur_R.copy()
    for i in range(3):
        if np.dot(prev_R[:, i], R[:, i]) < 0:
            R[:, i] *= -1
    # also ensure result is right-handed
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    return R

# ---------- Main pipeline ----------
df = load_df(INPUT_CSV)
n_frames = len(df)

# load groups (frames, markers, xyz)
torso = collect_group(df, TORSO_MARKERS)
upper = collect_group(df, UPPER_MARKERS)
forearm = collect_group(df, FOREARM_MARKERS)
hand = collect_group(df, HAND_MARKERS)

groups = {
    "torso": torso,
    "upper": upper,
    "forearm": forearm,
    "hand": hand
}

# Precompute orientations and centroids for all frames
orientations = {k: [] for k in groups}
centroids = {k: [] for k in groups}

# We'll also compute a "stick" chain: use centroid positions of blocks.
for i in range(n_frames):
    for name, arr in groups.items():
        if arr is None:
            R, c = np.eye(3), np.array([np.nan, np.nan, np.nan])
        else:
            pts = arr[i]  # shape: (markers, 3)
            R, c = svd_orientation(pts)
        orientations[name].append(R)
        centroids[name].append(c)

# Enforce axis continuity across time (prevent flipping arrow directions)
for name in orientations:
    prev = None
    fixed = []
    for R in orientations[name]:
        Rf = ensure_continuity(prev, R)
        fixed.append(Rf)
        prev = Rf
    orientations[name] = fixed
    centroids[name] = np.vstack(centroids[name])

# ---------- Create animation ----------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Autoscale based on all centroids
all_cent = np.vstack([centroids[name] for name in centroids if centroids[name] is not None])
# if all_nan then set arbitrary limits
if np.isnan(all_cent).all():
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
else:
    xmin, ymin, zmin = np.nanmin(all_cent, axis=0) - ARROW_SCALE*1.5
    xmax, ymax, zmax = np.nanmax(all_cent, axis=0) + ARROW_SCALE*1.5
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(zmin, zmax)

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
if SHOW_AXIS:
    ax.quiver(0,0,0,1,0,0,length=ARROW_SCALE*0.7,color='k')
    ax.quiver(0,0,0,0,1,0,length=ARROW_SCALE*0.7,color='k')
    ax.quiver(0,0,0,0,0,1,length=ARROW_SCALE*0.7,color='k')

# Artists
stick_line, = ax.plot([], [], [], 'o-', lw=3, color='black', label='stick')
torso_scatter = ax.scatter([],[],[], s=20, color='blue', label='torso markers')
upper_scatter = ax.scatter([],[],[], s=20, color='red',  label='upper markers')
fore_scatter  = ax.scatter([],[],[], s=20, color='green', label='fore markers')
hand_scatter  = ax.scatter([],[],[], s=20, color='purple',label='hand markers')

# quiver placeholders: we'll create and remove them each update to avoid stale state
quivers = []

def update(frame):
    global quivers
    # remove old quivers
    for q in quivers:
        try:
            q.remove()
        except Exception:
            pass
    quivers = []

    # Draw per-group marker clouds (if available)
    for name, arr in groups.items():
        if arr is None:
            continue
        pts = arr[frame]  # (markers, 3)
        xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]
        if name == 'torso':
            torso_scatter._offsets3d = (xs, ys, zs)
        elif name == 'upper':
            upper_scatter._offsets3d = (xs, ys, zs)
        elif name == 'forearm':
            fore_scatter._offsets3d = (xs, ys, zs)
        elif name == 'hand':
            hand_scatter._offsets3d = (xs, ys, zs)

    # Stick figure: connect centroids in chain torso -> upper -> forearm -> hand
    chain_names = ['torso','upper','forearm','hand']
    chain_centroids = []
    for nm in chain_names:
        c = centroids[nm][frame] if nm in centroids else np.array([np.nan,np.nan,np.nan])
        chain_centroids.append(c)
    chain_centroids = np.array(chain_centroids)
    stick_line.set_data(chain_centroids[:,0], chain_centroids[:,1])
    stick_line.set_3d_properties(chain_centroids[:,2])

    # Draw orientation axes as 3 colored arrows for each block (x=red, y=green, z=blue)
    color_map = {'torso': 'blue', 'upper': 'red', 'forearm': 'green', 'hand': 'purple'}
    for name in chain_names:
        R = orientations[name][frame]
        c = centroids[name][frame]
        if np.isnan(c).any():
            continue
        # each column of R is an axis vector in world coords
        vx, vy, vz = R[:,0], R[:,1], R[:,2]
        # draw three small arrows (use different color shades)
        qx = ax.quiver(c[0], c[1], c[2], vx[0], vx[1], vx[2], length=ARROW_SCALE, normalize=True, color=color_map[name])
        qy = ax.quiver(c[0], c[1], c[2], vy[0], vy[1], vy[2], length=ARROW_SCALE*0.8, normalize=True, color='gray')
        qz = ax.quiver(c[0], c[1], c[2], vz[0], vz[1], vz[2], length=ARROW_SCALE*0.6, normalize=True, color='black')
        quivers.extend([qx, qy, qz])

    ax.set_title(f"Frame {frame+1}/{n_frames}")
    return [stick_line, torso_scatter, upper_scatter, fore_scatter, hand_scatter] + quivers

anim = FuncAnimation(fig, update, frames=n_frames, interval=FRAME_INTERVAL_MS, blit=False)

# If running in a script, save to mp4 (requires ffmpeg). If in Jupyter, show via HTML.
def try_save():
    try:
        writer = FFMpegWriter(fps=1000/FRAME_INTERVAL_MS)
        anim.save(OUTPUT_MOV, writer=writer)
        print("Saved animation to", OUTPUT_MOV)
    except Exception as e:
        print("Could not save MP4 (ffmpeg missing or error):", e)

# Display / save:
if __name__ == "__main__":
    # running as script
    try_save()
else:
    # In Jupyter: display inline (this will embed a javascript animation)
    from IPython.display import HTML
    display(HTML(anim.to_jshtml()))
