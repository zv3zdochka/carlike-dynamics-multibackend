# control_method_3d_gif.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, writers
import torch
import seaborn as sns

if torch.cuda.is_available():
    import computations_torch as comp
else:
    import computations_cpu as comp

x_start, x_finish, nx = -5, 5, 101
y_start, y_finish, ny = -5, 5, 101
z_start, z_finish, nz = -10, 10, 101

t_start, t_finish, nt = 0, 3 * np.pi / 2, 31

u_min, u_max = -comp.V_ / comp.L_ * np.tan(comp.PHI_MAX), comp.V_ / comp.L_ * np.tan(comp.PHI_MAX)

x = np.linspace(x_start, x_finish, nx)
dx = x[1] - x[0]
y = np.linspace(y_start, y_finish, ny)
dy = y[1] - y[0]
z = np.linspace(z_start, z_finish, nz)
dz = z[1] - z[0]
t = np.linspace(t_start, t_finish, nt)
dt = t[1] - t[0]

start_p = np.array([[0, 0, np.pi / 2]]).T


def draw(list_of_pts, title='test'):
    xx, yy, zz = [], [], []
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    facecolors = []

    for i, points in enumerate(list_of_pts):
        for a in points:
            xx.append(a[0])
            yy.append(a[1])
            zz.append(a[2])
            facecolors.append(colors[i])

    xticks = []
    xticklabels = []
    for i, x_ in enumerate(x):
        if i % 50 == 0:
            xticks.append(i)
            xticklabels.append(np.round(x_, 1))

    yticks = []
    yticklabels = []
    for i, y_ in enumerate(y):
        if i % 50 == 0:
            yticks.append(i)
            yticklabels.append(np.round(y_, 1))

    zticks = []
    zticklabels = []
    for i, z_ in enumerate(z):
        if i % 50 == 0:
            zticks.append(i)
            zticklabels.append(np.round(z_, 1))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    ax.scatter(xx, yy, zz, facecolors=facecolors, edgecolors='black', marker='s')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_zticks(zticks)
    ax.set_zticklabels(zticklabels)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.title(title)

    def update(frame):
        ax.view_init(elev=frame, azim=frame)
        return fig,

    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=False)
    ani.save(title + '.gif', fps=30, writer='pillow')
    plt.show()


controls = [
    (u_min, 0, u_min),
    (u_min, 0, u_max),
    (u_max, 0, u_min),
    (u_max, 0, u_max),
    (u_min, u_max, u_min),
    (u_max, u_min, u_max)
]

list_of_points = comp.solve(start_p, controls, method='RK4', t=t, dt=dt, x_start=x_start, dx=dx, nx=nx, y_start=y_start,
                            dy=dy, ny=ny, z_start=z_start, dz=dz, nz=nz)
draw(list_of_points, title='The Runge-Kutta method 4th order')

list_of_points = comp.solve(start_p, controls, method='RK2', t=t, dt=dt, x_start=x_start, dx=dx, nx=nx, y_start=y_start,
                            dy=dy, ny=ny, z_start=z_start, dz=dz, nz=nz)
draw(list_of_points, title='The Runge-Kutta method 2th order')

list_of_points = comp.solve(start_p, controls, method='Euler', t=t, dt=dt, x_start=x_start, dx=dx, nx=nx,
                            y_start=y_start, dy=dy, ny=ny, z_start=z_start, dz=dz, nz=nz)
draw(list_of_points, title='The Euler method')
