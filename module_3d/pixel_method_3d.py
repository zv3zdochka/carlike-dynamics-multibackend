# pixel_method_3d.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import computations_cpu as comp_cpu
import computations_torch as comp_torch
import computations_pycuda as comp_pycuda

import torch  # оставляю импорт как в исходнике

x_start, x_finish, nx = -5, 5, 301
y_start, y_finish, ny = -5, 5, 301
z_start, z_finish, nz = -10, 10, 51

t_start, t_finish, nt = 0, 3 * np.pi / 2, 11
u_start, u_finish, nu = -comp_cpu.V_ / comp_cpu.L_ * np.tan(comp_cpu.PHI_MAX), comp_cpu.V_ / comp_cpu.L_ * np.tan(
    comp_cpu.PHI_MAX), 100

x = np.linspace(x_start, x_finish, nx)
dx = x[1] - x[0]
y = np.linspace(y_start, y_finish, ny)
dy = y[1] - y[0]
z = np.linspace(z_start, z_finish, nz)
dz = z[1] - z[0]
t = np.linspace(t_start, t_finish, nt)
dt = t[1] - t[0]

u = np.linspace(u_start, u_finish, nu)
du = u[1] - u[0]
all_u = comp_cpu.get_all_u(u)

m0 = np.array([[0, 0, np.pi / 2]]).T


def draw2d(q, title=''):
    q = np.rot90(q, 3)
    xticks = []
    xticklabels = []
    for i, x_ in enumerate(x):
        if i % 50 == 0:
            xticks.append(i)
            xticklabels.append(round(x_, 1))

    yticks = []
    yticklabels = []
    for i, y_ in enumerate(y):
        if i % 50 == 0:
            yticks.append(i)
            yticklabels.append(round(y_, 1))
    ax = sns.heatmap(q, cbar=False, square=True)
    ax.set(ylim=ax.get_ylim()[::-1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    plt.title(title)
    plt.show()


def draw(m, title=''):
    xx, yy, zz = [], [], []
    for a in m:
        xx.append(a[0])
        yy.append(a[1])
        zz.append(a[2])

    xticks = []
    xticklabels = []
    for i, x_ in enumerate(x):
        if i % 50 == 0:
            xticks.append(i)
            xticklabels.append(round(x_, 1))

    yticks = []
    yticklabels = []
    for i, y_ in enumerate(y):
        if i % 50 == 0:
            yticks.append(i)
            yticklabels.append(round(y_, 1))

    zticks = []
    zticklabels = []
    for i, z_ in enumerate(z):
        if i % 50 == 0:
            zticks.append(i)
            zticklabels.append(round(z_, 1))

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(xx, yy, zz, facecolors='white', edgecolors='black')
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
    plt.show()


times = {"RK4": {}, "RK2": {}, "Euler": {}}

print("Running CPU version for RK4")
q_cpu, m_cpu, sm_cpu, time_cpu = comp_cpu.solve_pixel(
    method='RK4', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
print(q_cpu.sum())
draw(sm_cpu, title='The Runge-Kutta method 4th order (CPU)')
times["RK4"]["CPU"] = time_cpu

print("Running TORCH version for RK4")
q_torch, m_torch, sm_torch, time_torch = comp_torch.solve_pixel(
    method='RK4', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
print(q_torch.sum())
draw(sm_torch, title='The Runge-Kutta method 4th order (TORCH)')
times["RK4"]["TORCH"] = time_torch

print("Running PYCUDA version for RK4")
q_pyc, m_pyc, sm_pyc, time_pyc = comp_pycuda.solve_pixel(
    method='RK4', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
print(q_pyc.sum())
draw(sm_pyc, title='The Runge-Kutta method 4th order (PYCUDA)')
times["RK4"]["PYCUDA"] = time_pyc

print(f"RK4 times: CPU {time_cpu}, TORCH {time_torch}, PYCUDA {time_pyc}")

# ---------- RK2 ----------
print("Running CPU version for RK2")
q_cpu, m_cpu, sm_cpu, time_cpu = comp_cpu.solve_pixel(
    method='RK2', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
draw(sm_cpu, title='The Runge-Kutta method 2th order (CPU)')
times["RK2"]["CPU"] = time_cpu

print("Running TORCH version for RK2")
q_torch, m_torch, sm_torch, time_torch = comp_torch.solve_pixel(
    method='RK2', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
draw(sm_torch, title='The Runge-Kutta method 2th order (TORCH)')
times["RK2"]["TORCH"] = time_torch

print("Running PYCUDA version for RK2")
q_pyc, m_pyc, sm_pyc, time_pyc = comp_pycuda.solve_pixel(
    method='RK2', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
draw(sm_pyc, title='The Runge-Kutta method 2th order (PYCUDA)')
times["RK2"]["PYCUDA"] = time_pyc

print(f"RK2 times: CPU {time_cpu}, TORCH {time_torch}, PYCUDA {time_pyc}")

# ---------- Euler ----------
print("Running CPU version for Euler")
q_cpu, m_cpu, sm_cpu, time_cpu = comp_cpu.solve_pixel(
    method='Euler', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
draw(sm_cpu, title='The Euler method (CPU)')
times["Euler"]["CPU"] = time_cpu

print("Running TORCH version for Euler")
q_torch, m_torch, sm_torch, time_torch = comp_torch.solve_pixel(
    method='Euler', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
draw(sm_torch, title='The Euler method (TORCH)')
times["Euler"]["TORCH"] = time_torch

print("Running PYCUDA version for Euler")
q_pyc, m_pyc, sm_pyc, time_pyc = comp_pycuda.solve_pixel(
    method='Euler', t=t, dt=dt, all_u=all_u, start_p=m0,
    x_start=x_start, dx=dx, nx=nx,
    y_start=y_start, dy=dy, ny=ny,
    z_start=z_start, dz=dz, nz=nz
)
draw(sm_pyc, title='The Euler method (PYCUDA)')
times["Euler"]["PYCUDA"] = time_pyc

print(f"Euler times: CPU {time_cpu}, TORCH {time_torch}, PYCUDA {time_pyc}")

methods = ["RK4", "RK2", "Euler"]
backends = ["CPU", "TORCH", "PYCUDA"]

cpu_vals = [times[m]["CPU"] for m in methods]
torch_vals = [times[m]["TORCH"] for m in methods]
pycuda_vals = [times[m]["PYCUDA"] for m in methods]

x_idx = np.arange(len(methods))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x_idx - width, cpu_vals, width, label='CPU')
plt.bar(x_idx, torch_vals, width, label='TORCH')
plt.bar(x_idx + width, pycuda_vals, width, label='PYCUDA')

plt.xticks(x_idx, methods)
plt.ylabel('Time, s')
plt.title('Time comparison: CPU vs TORCH vs PYCUDA')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
