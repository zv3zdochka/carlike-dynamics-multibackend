# computations_cpu.py
import numpy as np


def to_backend(arr):
    return np.asarray(arr, dtype=np.float32, order="C")


def sync():
    return


def shape(x):
    return x.shape


def evaluate_field(X, u, V=1.0):
    X = np.asarray(X, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    theta = X[:, 2].astype(np.float32)
    dx = V * np.cos(theta).astype(np.float32)
    dy = V * np.sin(theta).astype(np.float32)
    dth = u
    out = np.empty_like(X, dtype=np.float32)
    out[:, 0] = dx
    out[:, 1] = dy
    out[:, 2] = dth
    return out


def euler_step(X, u, dt, V=1.0):
    return X + dt * evaluate_field(X, u, V=V)


def euler_rollout(X0, u, dt, steps, V=1.0):
    X = np.asarray(X0, dtype=np.float32).copy()
    for _ in range(steps):
        X += dt * evaluate_field(X, u, V=V)
    return X
