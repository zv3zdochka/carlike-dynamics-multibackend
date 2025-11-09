# computations_cpu.py
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
import seaborn as sns

V_ = 1
L_ = 1
PHI_MAX = np.radians(45)
LINE_INTERPOLATION = False


def get_points(ij_st, ij_fn):
    if LINE_INTERPOLATION:
        n = np.max(np.abs(ij_st - ij_fn))
        iijj = np.linspace(ij_st, ij_fn, n + 1, dtype=int, axis=0).T
        return iijj
    return ij_fn.reshape(-1, 1)


def foo(t_, x_, u_):
    f = [
        V_ * np.cos(x_[2]),
        V_ * np.sin(x_[2]),
        x_[2] * 0 + u_
    ]
    res = np.array(f)
    return res


def euler(t0, dt_, x_, u_, func):
    res = x_ + func(t0, x_, u_) * dt_
    return res


def runge_kutta_2(t0, dt_, x_, u_, func):
    res = x_ + func(t0, x_ + foo(t0 + dt_ / 2, x_, u_) * dt_ / 2, u_) * dt_
    return res


def runge_kutta_4(t0, dt_, x_, u_, func):
    k1 = func(t0, x_, u_)
    k2 = func(t0 + dt_ / 2, x_ + dt_ * k1 / 2, u_)
    k3 = func(t0 + dt_ / 2, x_ + dt_ * k2 / 2, u_)
    k4 = func(t0 + dt_, x_ + dt_ * k3, u_)
    res = x_ + dt_ * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return res


def ij_to_xy(ij, x_start, dx, y_start, dy, z_start, dz):
    return np.vstack([ij[0] * dx + x_start, ij[1] * dy + y_start, ij[2] * dz + z_start])


def xy_to_ij(xy, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz):
    return np.vstack([np.clip(np.round((xy[0] - x_start) / dx), 0, nx - 1),
                      np.clip(np.round((xy[1] - y_start) / dy), 0, ny - 1),
                      np.clip(np.round((xy[2] - z_start) / dz), 0, nz - 1)]).astype(int)


def solve_control(start_point, control, method='RK4', t=None, dt=None, x_start=None, dx=None, nx=None, y_start=None,
                  dy=None, ny=None, z_start=None, dz=None, nz=None):
    res = set()
    for i in range(t.shape[0]):
        for j in range(i, t.shape[0]):
            xx_new = start_point
            for k in range(t.shape[0]):
                if k <= i:
                    u_ = control[0]
                elif k <= j:
                    u_ = control[1]
                else:
                    u_ = control[2]

                if method == 'RK4':
                    xx_new = runge_kutta_4(t[k], dt, xx_new, u_, foo)
                elif method == 'RK2':
                    xx_new = runge_kutta_2(t[k], dt, xx_new, u_, foo)
                elif method == 'Euler':
                    xx_new = euler(t[k], dt, xx_new, u_, foo)
                else:
                    raise NotImplementedError

            ij_new = xy_to_ij(xx_new, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
            res.add((ij_new[0, 0], ij_new[1, 0], ij_new[2, 0]))
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
    return np.array(res)


def solve_pixel(method='RK4', t=None, dt=None, all_u=None, start_p=None, x_start=None, dx=None, nx=None, y_start=None,
                dy=None, ny=None, z_start=None, dz=None, nz=None):
    time_start = perf_counter()
    idx = xy_to_ij(start_p, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
    m = set()
    for i in range(idx.shape[1]):
        m.add((idx[0, i], idx[1, i], idx[2, i]))

    sm = m.copy()

    for time in t:
        if not sm:
            break

        m_new = set()
        ij_ = np.array(list(sm)).T
        xx_ = ij_to_xy(ij_, x_start, dx, y_start, dy, z_start, dz)
        for iu, u_ in enumerate(all_u):
            if method == 'RK4':
                xx_new = runge_kutta_4(time, dt, xx_, u_, foo)
                angle = xx_new[2]
                # angle = np.atan2(np.sin(angle), np.cos(angle))
                xx_new = np.array([xx_new[0], xx_new[1], angle])

                # xx_new_2 = runge_kutta_2(time, dt, xx_, u_, foo)
                # xx_new_e = euler(time, dt, xx_, u_, foo)
            elif method == 'RK2':
                xx_new = runge_kutta_2(time, dt, xx_, u_, foo)
            elif method == 'Euler':
                xx_new = euler(time, dt, xx_, u_, foo)
            else:
                raise NotImplementedError

            ij_new = xy_to_ij(xx_new, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)

            for r in range(ij_new.shape[1]):
                line = get_points(ij_[:, r], ij_new[:, r])
                for i in range(line.shape[1]):
                    m_new.add((line[0, i], line[1, i], line[2, i]))
        sm = m_new - m
        m = m.union(m_new)
        print(time, len(sm))

    q = np.zeros((nx, ny, nz))

    ij_ = tuple(np.array(list(m)).T)

    q[ij_] = 1
    time_spent = perf_counter() - time_start
    print(method, 'Time: ', time_spent)
    return q, m, sm, time_spent
