import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter

LINE_INTERPOLATION = False


def get_points(ij_st, ij_fn): #
    if LINE_INTERPOLATION:
        n = np.max(np.abs(ij_st - ij_fn))
        iijj = np.linspace(ij_st, ij_fn, n + 1, dtype=int, axis=0).T
        return iijj
    return ij_fn.reshape(-1, 1)


def get_all_u(u_):
    res = []
    for u1 in u_:
        for u2 in u_:
            if u1 ** 2 + 25 * u2 ** 2 <= 1:
                res.append([u1, u2])
    return np.array(res)


def foo(t_, x_, u_):
    res = np.array([
        -u_[1] * x_[0] + u_[0] * x_[1],
        -u_[0] * x_[0] - u_[1] * x_[1]
    ])
    return res


def euler(t0, dt, x_, u_, func):
    res = x_ + func(t0, x_, u_) * dt
    return res


def runge_kutta_2(t0, dt, x_, u_, func):
    res = x_ + func(t0, x_ + foo(t0 + dt / 2, x_, u_) * dt / 2, u_) * dt
    return res


def runge_kutta_4(t0, dt, x_, u_, func):
    k1 = func(t0, x_, u_)
    k2 = func(t0 + dt / 2, x_ + dt * k1 / 2, u_)
    k3 = func(t0 + dt / 2, x_ + dt * k2 / 2, u_)
    k4 = func(t0 + dt, x_ + dt * k3, u_)
    res = x_ + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return res


def ij_to_xy(ij):
    # return np.vstack([x[ij[0]], y[ij[1]]])
    return np.vstack([ij[0] * dx + x_start, ij[1] * dy + y_start])


def xy_to_ij(xy):
    return np.vstack([np.clip(np.round((xy[0] - x_start) / dx), 0, x.shape[0] - 1),
                      np.clip(np.round((xy[1] - y_start) / dy), 0, y.shape[0] - 1)]).astype(int)


time_start = perf_counter()

x_start, x_finish, dx = -2, 2, 0.0083
y_start, y_finish, dy = -2, 2, 0.0083
t_start, t_finish, dt = 0, 2.7, 0.3
u_start, u_finish, du = -1, 1, 0.05

x = np.arange(x_start, x_finish + dx, dx)
# x = np.linspace(x_start, x_finish, 500)

y = np.arange(y_start, y_finish + dy, dy)
# y = np.linspace(y_start, y_finish, 500)
t = np.arange(t_start, t_finish + dt, dt)

u = np.arange(u_start, u_finish + du, du)
all_u = get_all_u(u)

m0 = np.array([[1, 0]]).T


def solve(method='RK4'):
    idx = xy_to_ij(m0)
    m = set()
    for i in range(idx.shape[1]):
        m.add((idx[0, i], idx[1, i]))

    sm = m.copy()

    for time in t:
        m_new = set()
        ij_ = np.array(list(sm)).T
        xx_ = ij_to_xy(ij_)
        for iu, u_ in enumerate(all_u):
            if method == 'RK4':
                xx_new = runge_kutta_4(time, dt, xx_, u_, foo)
                xx_new_2 = runge_kutta_2(time, dt, xx_, u_, foo)
                xx_new_e = euler(time, dt, xx_, u_, foo)
            elif method == 'RK2':
                xx_new = runge_kutta_2(time, dt, xx_, u_, foo)
            elif method == 'Euler':
                xx_new = euler(time, dt, xx_, u_, foo)
            else:
                raise NotImplementedError

            ij_new = xy_to_ij(xx_new)

            for r in range(ij_new.shape[1]):
                line = get_points(ij_[:, r], ij_new[:, r])
                for i in range(line.shape[1]):
                    m_new.add((line[0, i], line[1, i]))
        sm = m_new - m
        m = m.union(m_new)
        print(time, len(sm))

    q = np.zeros((y.shape[0], x.shape[0]))

    ij_ = tuple(np.array(list(m)).T)

    q[ij_] = 1
    q = q.T
    return q


q = solve()
print('Время: ', perf_counter() - time_start)


def draw(q, title=''):
    xticklabels = ['' if i % 20 != 0 else np.round(x[i], 3) for i in range(x.shape[0])]
    yticklabels = ['' if i % 20 != 0 else np.round(y[i], 3) for i in range(y.shape[0])]
    fig = sns.heatmap(q, xticklabels=xticklabels, yticklabels=yticklabels, cbar=False, square=True)
    fig.set(ylim=fig.get_ylim()[::-1])
    plt.title(title)
    plt.show()


draw(q, title='Метод Рунге-Кутты 4 порядка')

q = solve('RK2')
draw(q, title='Метод Рунге-Кутты 2 порядка')

q = solve('Euler')
draw(q, title='Метод Эйлера')
