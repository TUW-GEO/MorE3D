import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

if __name__ == '__main__':

    fig = plt.figure()
    fig2= plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax2 = fig2.add_subplot(projection='3d')
    X = np.arange(-15, 15, 1)
    Y = np.arange(-15, 15, 1)
    X, Y = np.meshgrid(X, Y)
    # plane
    Z = np.ones(np.shape(X))

    # Plot the plane.
    scat = ax.scatter(X, Y, Z, cmap=cm.coolwarm, marker='.',
                           linewidth=0, antialiased=False)

    surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # sinus
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    Z = np.sin(X/6)
    surf2 = ax3.scatter(X, np.abs(Y), Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax3.set_box_aspect([1,1,1])
    set_axes_equal(ax3)  # IMPORTANT - this is also required
    plt.show()

