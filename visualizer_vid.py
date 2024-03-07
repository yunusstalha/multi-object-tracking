import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from scipy.stats.distributions import chi2


tracks = np.load("tracks.npy", allow_pickle=True)
measurements = np.load("measurements.npy", allow_pickle=True)



P_DET = 0.95
P_GATE = 0.997
BETA_FA = 10 / (1000 ** 2) 
BETA_NT = 3 / 350 / (1000 ** 2)
GATE_THRESHOLD = np.sqrt(chi2.ppf(P_GATE, df=4)) 
num_obj = []
for i in range(350):
    if i < 50:
        num_obj.append(1)
    elif i < 100:
        num_obj.append(2)
    elif i < 250:
        num_obj.append(3)
    else:
        num_obj.append(1)

print(GATE_THRESHOLD)
# print(tracks[1][:, 0])
# print(measurements[50].shape) 
fig, ax = plt.subplots(figsize=(24, 14))
ax.set_xlim((-1000, 1000))
ax.set_ylim((-1000, 1000))

def update(time):
    ax.clear()
    ax.set_xlim((-1000, 1000)), ax.set_ylim((-1000, 1000))

    major_ticks = np.arange(-1000, 1000, 200)
    minor_ticks = np.arange(-1000, 1000, 50)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


    for measurement in range(measurements[time].shape[0]):
        x, y, w, h = measurements[time][measurement, :4]
        x, y = x - w / 2, y - h / 2
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    for track in tracks:
        for track_id in range(track.shape[0]):
            if track[track_id, 0] == time:
                x, y, w, h = track[track_id, 1], track[track_id, 3], track[track_id, 5], track[track_id, 6]
                center = np.array([x, y])
                x, y = x - w / 2, y - h / 2
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

                innovation_covariance = track[track_id, 7:].reshape(4, 4)[:2, :2]
                eigvals, eigvecs = np.linalg.eigh(innovation_covariance)
                radii = GATE_THRESHOLD * np.sqrt(eigvals)
                angle = np.degrees(np.arctan2(*eigvecs[:, 1]))
                ellipse = patches.Ellipse(center, width=radii[0] * 2, height=radii[1] * 2, angle=angle, color='b', alpha=0.2)
                ax.add_patch(ellipse)

    # Update plot labels and legend
    title = r"$ Time: $" + str(time) + " " + r'$ \beta_{FA} $: ' + "{:.3e}".format(BETA_FA) + " " + r'$ \beta_{NT} $: ' + "{:.3e}".format(BETA_NT) + " " + r'$ P_{D} $: ' + str(P_DET) + " " + r'$ P_{G} $: ' + str(P_GATE)
    ax.set_title(title, fontsize=20)
    red_patch = patches.Patch(color='red', label='Measurements')
    blue_patch = patches.Patch(color='blue', label='Tracked Targets')
    ax.legend(handles=[red_patch, blue_patch])
    ax.set_xlabel('X (meters)', fontsize=15)
    ax.set_ylabel('Y (meters)', fontsize=15)

# Create the animation
num_frames = measurements.shape[0]  # Adjust according to your data
ani = FuncAnimation(fig, update, frames=range(num_frames), blit=False)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# Save the animation
ani.save('exp5/tracking_animation.mp4', writer=writer)