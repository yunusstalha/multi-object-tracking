import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example data - replace this with your actual data
# Assuming it's a 700x4 numpy array (700 time steps with 4D data each)
data = np.load("/home/yunusi/git/multi-object-tracking/measurements.npy", allow_pickle=True) 

# Set up the figure and axis
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'rx')

def init():
    ax.set_xlim(-3000, 3000)  # Set these limits according to your data
    ax.set_ylim(-3000, 3000)  # Set these limits according to your data
    return ln,

def update(frame):
    print(data[frame].shape)
    xdata.append(data[frame][:,0])
    ydata.append(data[frame][:,1])
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=range(700),
                    init_func=init, blit=True)

# To save the animation, uncomment the following line:
# ani.save('animation.mp4', writer='ffmpeg')

plt.show()
