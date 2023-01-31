import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-15, 15, 0.25)
    Y = np.arange(-15, 15, 0.25)
    X, Y = np.meshgrid(X, Y)
    rho = 10
    sigma = 1

    Z = 1/(2*sigma*np.pi) * np.exp(-0.5 * (np.sqrt(X**2+Y**2)-rho)/sigma)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()