# computations_cuda.py
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

V_ = 1.0
L_ = 1.0
PHI_MAX = np.radians(45)
LINE_INTERPOLATION = False


def get_points(ij_st, ij_fn):
    if LINE_INTERPOLATION:
        n = torch.max(torch.abs(ij_st - ij_fn))
        iijj = torch.linspace(ij_st, ij_fn, n + 1, dtype=torch.int32).T
        return iijj
    return ij_fn.reshape(-1, 1)


def foo(t_, x_, u_):
    res = torch.stack([
        V_ * torch.cos(x_[2]),
        V_ * torch.sin(x_[2]),
        torch.zeros_like(x_[2]) + u_
    ], dim=0)
    return res


def euler(t0, dt_, x_, u_, func):
    res = x_ + func(t0, x_, u_) * dt_
    return res


def runge_kutta_2(t0, dt_, x_, u_, func):
    res = x_ + func(t0, x_ + foo(t0 + dt_ / 2, x_, u_) * (dt_ / 2), u_) * dt_
    return res


def runge_kutta_4(t0, dt_, x_, u_, func):
    k1 = func(t0, x_, u_)
    k2 = func(t0 + (dt_ / 2), x_ + dt_ * k1 / 2, u_)
    k3 = func(t0 + (dt_ / 2), x_ + dt_ * k2 / 2, u_)
    k4 = func(t0 + dt_, x_ + dt_ * k3, u_)
    res = x_ + dt_ * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return res


def ij_to_xy(ij, x_start, dx, y_start, dy, z_start, dz):
    dev = ij.device
    dtype = torch.float32
    x_start = torch.tensor(x_start, device=dev, dtype=dtype)
    dx = torch.tensor(dx, device=dev, dtype=dtype)
    y_start = torch.tensor(y_start, device=dev, dtype=dtype)
    dy = torch.tensor(dy, device=dev, dtype=dtype)
    z_start = torch.tensor(z_start, device=dev, dtype=dtype)
    dz = torch.tensor(dz, device=dev, dtype=dtype)
    return torch.stack([ij[0].to(dtype) * dx + x_start, ij[1].to(dtype) * dy + y_start, ij[2].to(dtype) * dz + z_start],
                       dim=0)


def xy_to_ij(xy, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz):
    dev = xy.device
    dtype = xy.dtype
    x_start = torch.tensor(x_start, device=dev, dtype=dtype)
    dx = torch.tensor(dx, device=dev, dtype=dtype)
    y_start = torch.tensor(y_start, device=dev, dtype=dtype)
    dy = torch.tensor(dy, device=dev, dtype=dtype)
    z_start = torch.tensor(z_start, device=dev, dtype=dtype)
    dz = torch.tensor(dz, device=dev, dtype=dtype)
    i_x = torch.clamp(torch.round((xy[0] - x_start) / dx), 0, nx - 1).to(torch.int32)
    i_y = torch.clamp(torch.round((xy[1] - y_start) / dy), 0, ny - 1).to(torch.int32)
    i_z = torch.clamp(torch.round((xy[2] - z_start) / dz), 0, nz - 1).to(torch.int32)
    return torch.stack([i_x, i_y, i_z], dim=0)


def solve_control(start_point, control, method='RK4', t=None, dt=None, x_start=None, dx=None, nx=None, y_start=None,
                  dy=None, ny=None, z_start=None, dz=None, nz=None):
    start_point = torch.as_tensor(start_point, device=device, dtype=torch.float32)
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    dt = torch.tensor(dt, device=device, dtype=torch.float32)
    res = set()
    for i in range(t.shape[0]):
        for j in range(i, t.shape[0]):
            xx_new = start_point.clone()
            for k in range(t.shape[0]):
                if k <= i:
                    u_ = control[0]
                elif k <= j:
                    u_ = control[1]
                else:
                    u_ = control[2]
                u_ = torch.tensor(float(u_), device=device, dtype=torch.float32)
                if method == 'RK4':
                    xx_new = runge_kutta_4(t[k], dt, xx_new, u_, foo)
                elif method == 'RK2':
                    xx_new = runge_kutta_2(t[k], dt, xx_new, u_, foo)
                elif method == 'Euler':
                    xx_new = euler(t[k], dt, xx_new, u_, foo)
                else:
                    raise NotImplementedError
            ij_new = xy_to_ij(xx_new, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
            res.add((ij_new[0, 0].item(), ij_new[1, 0].item(), ij_new[2, 0].item()))
    return res


def solve(start_point, controls_, method='RK4', t=None, dt=None, x_start=None, dx=None, nx=None, y_start=None, dy=None,
          ny=None, z_start=None, dz=None, nz=None):
    time_start = perf_counter()
    list_of_pts = []
    for control in controls_:
        points = solve_control(start_point, control=control, method=method, t=t, dt=dt, x_start=x_start, dx=dx, nx=nx,
                               y_start=y_start, dy=dy, ny=ny, z_start=z_start, dz=dz, nz=nz)
        list_of_pts.append(points)
    time_spent = perf_counter() - time_start
    print(f"{method} Time: {time_spent}")
    return list_of_pts


def get_all_u(u_):
    res = []
    for u1 in u_:
        res.append(u1)
    return torch.tensor(res, device=device)


def solve_pixel(method='RK4', t=None, dt=None, all_u=None, start_p=None, x_start=None, dx=None, nx=None, y_start=None,
                dy=None, ny=None, z_start=None, dz=None, nz=None):
    time_start = perf_counter()
    start_p = torch.as_tensor(start_p, device=device, dtype=torch.float32)
    dt = torch.tensor(dt, device=device, dtype=torch.float32)
    all_u = torch.as_tensor(all_u, device=device, dtype=torch.float32)
    t = torch.as_tensor(t, device=device, dtype=torch.float32)
    idx = xy_to_ij(start_p, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
    visited = torch.zeros((nx, ny, nz), dtype=torch.bool, device=device)
    visited[idx[0, 0], idx[1, 0], idx[2, 0]] = True
    sm_idx = torch.nonzero(visited).T.to(torch.int32)
    for time_idx in range(t.shape[0]):
        time = t[time_idx]
        if sm_idx.shape[1] == 0:
            break
        time_t = torch.tensor(time, device=device, dtype=torch.float32)
        old_visited = visited.clone()
        xx_ = ij_to_xy(sm_idx, x_start, dx, y_start, dy, z_start, dz)
        num_points = xx_.shape[1]
        xx_new_all = torch.zeros((3, num_points * all_u.shape[0]), device=device, dtype=torch.float32)
        for iu in range(all_u.shape[0]):
            u_ = all_u[iu]
            if method == 'RK4':
                xx_new = runge_kutta_4(time_t, dt, xx_, u_, foo)
                angle = xx_new[2]
                xx_new = torch.stack([xx_new[0], xx_new[1], angle], dim=0)
            elif method == 'RK2':
                xx_new = runge_kutta_2(time_t, dt, xx_, u_, foo)
            elif method == 'Euler':
                xx_new = euler(time_t, dt, xx_, u_, foo)
            else:
                raise NotImplementedError
            start = iu * num_points
            end = start + num_points
            xx_new_all[:, start:end] = xx_new
        ij_new_all = xy_to_ij(xx_new_all, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
        if LINE_INTERPOLATION:
            m_new = set()
            for r in range(ij_new_all.shape[1]):
                line = get_points(sm_idx[:, r % num_points], ij_new_all[:, r])
                for i in range(line.shape[1]):
                    m_new.add((line[0, i].item(), line[1, i].item(), line[2, i].item()))
            for pt in m_new:
                visited[pt[0], pt[1], pt[2]] = True
        else:
            visited[ij_new_all[0, :], ij_new_all[1, :], ij_new_all[2, :]] = True
        new_mask = visited & ~old_visited
        sm_idx = torch.nonzero(new_mask).T.to(torch.int32)
        print(time.item(), sm_idx.shape[1])
    q = visited.to(torch.float32).cpu().numpy()
    all_idx = torch.nonzero(visited).cpu().numpy()
    m = set(tuple(row) for row in all_idx)
    last_idx = torch.nonzero(new_mask).cpu().numpy()
    sm = set(tuple(row) for row in last_idx)
    time_spent = perf_counter() - time_start
    print(method, 'Time: ', time_spent)
    return q, m, sm, time_spent
