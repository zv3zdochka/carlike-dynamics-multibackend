# control_main.py
from time import perf_counter
import importlib
import importlib.util
import sys
import os
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------- Бэкенды -----------------------------------------

BACKENDS = [
    ("CPU", "computations_cpu"),
    ("TORCH", "computations_torch"),
    ("PYCUDA", "computations_pycuda"),
]


def bench(fn, sync=None, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
        sync and sync()
    times = []
    for _ in range(repeat):
        t0 = perf_counter()
        fn()
        sync and sync()
        times.append(perf_counter() - t0)
    return min(times)


def _euler_step_generic(mod, X, u, dt, V):
    if hasattr(mod, "euler_step"):
        return mod.euler_step(X, u, dt, V=V)
    return X + dt * mod.evaluate_field(X, u, V=V)


def _rk2_step_generic(mod, X, u, dt, V):
    k1 = mod.evaluate_field(X, u, V=V)
    X_mid = X + (0.5 * dt) * k1
    k2 = mod.evaluate_field(X_mid, u, V=V)
    return X + dt * k2


def _rk4_step_generic(mod, X, u, dt, V):
    k1 = mod.evaluate_field(X, u, V=V)
    k2 = mod.evaluate_field(X + 0.5 * dt * k1, u, V=V)
    k3 = mod.evaluate_field(X + 0.5 * dt * k2, u, V=V)
    k4 = mod.evaluate_field(X + dt * k3, u, V=V)
    return X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _copy_like(x):
    if hasattr(x, "clone"):
        return x.clone()
    if hasattr(x, "copy"):
        return x.copy()
    return x + 0


def _rollout_generic(mod, step_fn, X0, u, dt, steps, V):
    X = _copy_like(X0)
    for _ in range(int(steps)):
        X = step_fn(mod, X, u, dt, V)
    return X


def _rollout_euler(mod, X0, u, dt, steps, V):
    if hasattr(mod, "euler_rollout"):
        return mod.euler_rollout(X0, u, dt, steps, V=V)
    return _rollout_generic(mod, _euler_step_generic, X0, u, dt, steps, V)


def import_local_fresh(module_name: str):
    here = Path(__file__).resolve().parent
    candidate = here / f"{module_name}.py"
    sys.modules.pop(module_name, None)
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(module_name, str(candidate))
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(module_name)


# ---------------------- ВИЗУАЛИЗАЦИЯ РЕШЕНИЙ (heatmap) -----------------------
# Реализация по мотивам твоего кода с foo / сеткой и перебором управлений.
# Здесь всё чисто на NumPy, бэкенды не трогаем. Графики строятся по методам,
# а в заголовке добавляем имя бэкенда, чтобы получить 3 * nb картинок.

LINE_INTERPOLATION = False

# Параметры сетки и времени (как в примере)
HM_X_START, HM_X_FINISH, HM_DX = -2.0, 2.0, 0.0083
HM_Y_START, HM_Y_FINISH, HM_DY = -2.0, 2.0, 0.0083
HM_T_START, HM_T_FINISH, HM_DT = 0.0, 2.7, 0.3
HM_U_START, HM_U_FINISH, HM_DU = -1.0, 1.0, 0.05

HM_X = np.arange(HM_X_START, HM_X_FINISH + HM_DX, HM_DX)
HM_Y = np.arange(HM_Y_START, HM_Y_FINISH + HM_DY, HM_DY)
HM_T = np.arange(HM_T_START, HM_T_FINISH + HM_DT, HM_DT)
HM_U = np.arange(HM_U_START, HM_U_FINISH + HM_DU, HM_DU)


def hm_get_points(ij_st, ij_fn):
    if LINE_INTERPOLATION:
        n = np.max(np.abs(ij_st - ij_fn))
        iijj = np.linspace(ij_st, ij_fn, n + 1, dtype=int, axis=0).T
        return iijj
    return ij_fn.reshape(-1, 1)


def hm_get_all_u(u_):
    res = []
    for u1 in u_:
        for u2 in u_:
            if u1 ** 2 + 25 * u2 ** 2 <= 1:
                res.append([u1, u2])
    return np.array(res)


HM_ALL_U = hm_get_all_u(HM_U)

# начальное множество (точка (1, 0))
HM_M0 = np.array([[1.0, 0.0]]).T


def hm_foo(t_, x_, u_):
    # x_: shape (2, M), u_: shape (2,)
    return np.array([
        -u_[1] * x_[0] + u_[0] * x_[1],
        -u_[0] * x_[0] - u_[1] * x_[1],
    ])


def hm_euler_step(t0, dt, x_, u_):
    return x_ + hm_foo(t0, x_, u_) * dt


def hm_rk2_step(t0, dt, x_, u_):
    # классический midpoint RK2
    k1 = hm_foo(t0, x_, u_)
    x_mid = x_ + 0.5 * dt * k1
    k2 = hm_foo(t0 + 0.5 * dt, x_mid, u_)
    return x_ + dt * k2


def hm_rk4_step(t0, dt, x_, u_):
    k1 = hm_foo(t0, x_, u_)
    k2 = hm_foo(t0 + 0.5 * dt, x_ + 0.5 * dt * k1, u_)
    k3 = hm_foo(t0 + 0.5 * dt, x_ + 0.5 * dt * k2, u_)
    k4 = hm_foo(t0 + dt, x_ + dt * k3, u_)
    return x_ + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def hm_ij_to_xy(ij):
    # ij: shape (2, M)
    return np.vstack([
        ij[0] * HM_DX + HM_X_START,
        ij[1] * HM_DY + HM_Y_START,
    ])


def hm_xy_to_ij(xy):
    # xy: shape (2, M)
    i = np.clip(np.round((xy[0] - HM_X_START) / HM_DX), 0, HM_X.shape[0] - 1)
    j = np.clip(np.round((xy[1] - HM_Y_START) / HM_DY), 0, HM_Y.shape[0] - 1)
    return np.vstack([i, j]).astype(int)


def hm_solve(method="RK4"):
    idx = hm_xy_to_ij(HM_M0)
    m = {(int(idx[0, i]), int(idx[1, i])) for i in range(idx.shape[1])}
    sm = m.copy()

    for t_curr in HM_T:
        m_new = set()
        if not sm:
            break
        ij_arr = np.array(list(sm)).T  # shape (2, M)
        xx = hm_ij_to_xy(ij_arr)

        for u_vec in HM_ALL_U:
            if method == "RK4":
                xx_new = hm_rk4_step(t_curr, HM_DT, xx, u_vec)
            elif method == "RK2":
                xx_new = hm_rk2_step(t_curr, HM_DT, xx, u_vec)
            elif method == "Euler":
                xx_new = hm_euler_step(t_curr, HM_DT, xx, u_vec)
            else:
                raise NotImplementedError(f"Unknown method {method}")

            ij_new = hm_xy_to_ij(xx_new)

            for r in range(ij_new.shape[1]):
                line = hm_get_points(ij_arr[:, r], ij_new[:, r])
                for k in range(line.shape[1]):
                    m_new.add((int(line[0, k]), int(line[1, k])))

        sm = m_new - m
        m |= m_new
        print(f"[VIS] t={t_curr:.2f}, new points={len(sm)}, total={len(m)}")

    q = np.zeros((HM_Y.shape[0], HM_X.shape[0]), dtype=np.float32)
    if m:
        ij_ = tuple(np.array(list(m)).T)
        q[ij_] = 1.0
    return q.T  # shape (len(HM_X), len(HM_Y))


def hm_draw(q, method, backend_name=None):
    xticklabels = [
        "" if i % 20 != 0 else np.round(HM_X[i], 3) for i in range(HM_X.shape[0])
    ]
    yticklabels = [
        "" if i % 20 != 0 else np.round(HM_Y[i], 3) for i in range(HM_Y.shape[0])
    ]
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(
        q,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar=False,
        square=True,
    )
    ax.set(ylim=ax.get_ylim()[::-1])
    title = f"Method {method}"
    if backend_name is not None:
        title += f" | backend: {backend_name}"
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.tight_layout()
    plt.show()


# ----------------------------- main -----------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--V", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_torch_cpu", action="store_true",
                    help="Force-disable CUDA in Torch (sets TORCH_FORCE_CPU=1 before import)")
    ap.add_argument("--disable_jit", action="store_true",
                    help="Disable Torch JIT (sets PYTORCH_JIT=0 before import)")
    ap.add_argument("--no_solution_plots", action="store_true",
                    help="Disable solution heatmaps (only timing + bar chart)")
    args = ap.parse_args()

    if args.force_torch_cpu:
        os.environ["TORCH_FORCE_CPU"] = "1"
    if args.disable_jit:
        os.environ["PYTORCH_JIT"] = "0"

    rng = np.random.default_rng(args.seed)
    N = args.n
    dt = float(args.dt)
    V = float(args.V)

    X0 = np.empty((N, 3), dtype=np.float32)
    X0[:, 0] = rng.uniform(-5, 5, size=N).astype(np.float32)
    X0[:, 1] = rng.uniform(-5, 5, size=N).astype(np.float32)
    X0[:, 2] = rng.uniform(-math.pi, math.pi, size=N).astype(np.float32)
    u = rng.uniform(-1.0, 1.0, size=N).astype(np.float32)

    print(f"\n[SETUP] N={N:,}, steps={args.steps}, dt={dt}, V={V}")

    methods = ["RK4", "RK2", "Euler"]
    times = {m: {} for m in methods}
    present_backends = []

    for backend_name, module_name in BACKENDS:
        try:
            mod = import_local_fresh(module_name)
            print(f"[PATH] {backend_name}: {getattr(mod, '__file__', '?')}")
        except Exception as e:
            print(f"[WARN] {backend_name}: module '{module_name}' not loaded: {e}")
            continue

        try:
            info = getattr(mod, "backend_info", lambda: backend_name)()
            print(f"[INIT] {backend_name}: {info}")
            sync = getattr(mod, "sync", None)
            Xb = mod.to_backend(X0)
            ub = mod.to_backend(u)

            te = bench(lambda: _rollout_euler(mod, Xb, ub, dt, args.steps, V), sync=sync)
            times["Euler"][backend_name] = te
            print(f"[OK] {backend_name:6s} | Euler rollout: {te:.4f}s")

            trk2 = bench(lambda: _rollout_generic(mod, _rk2_step_generic, Xb, ub, dt, args.steps, V), sync=sync)
            times["RK2"][backend_name] = trk2
            print(f"[OK] {backend_name:6s} | RK2   rollout: {trk2:.4f}s")

            trk4 = bench(lambda: _rollout_generic(mod, _rk4_step_generic, Xb, ub, dt, args.steps, V), sync=sync)
            times["RK4"][backend_name] = trk4
            print(f"[OK] {backend_name:6s} | RK4   rollout: {trk4:.4f}s")

            present_backends.append(backend_name)

        except Exception as e:
            print(f"[ERR] {backend_name}: benchmark failed: {e}")

    if not present_backends:
        print("\n[ERROR] No backend was successfully measured.")
        return

    print("\n=== Timing summary (seconds) by method ===")
    header = "Method    " + "  ".join(f"{b:>10s}" for b in present_backends)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:9s} " + "  ".join(
            f"{times[m].get(b, float('nan')):10.4f}" for b in present_backends
        )
        print(row)

    # --- столбчатый график времени (как было) ---
    x_idx = np.arange(len(methods))
    nb = len(present_backends)
    width = 0.8 / nb
    offsets = (np.arange(nb) - (nb - 1) / 2.0) * width

    plt.figure(figsize=(10, 5))
    for i, bname in enumerate(present_backends):
        vals = [times[m][bname] for m in methods]
        plt.bar(x_idx + offsets[i], vals, width, label=bname)

    plt.xticks(x_idx, methods)
    plt.ylabel('Time (s)')
    plt.title('Speed comparison: CPU vs TORCH vs PYCUDA (by methods)')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- дополнительные 2D-графики решений по мотивам foo-сценария ---
    if not args.no_solution_plots:
        print("\n[VIS] Computing solution heatmaps for each method...")
        heatmaps_cache = {}
        for m in methods:
            heatmaps_cache[m] = hm_solve(m)

        for bname in present_backends:
            for m in methods:
                hm_draw(heatmaps_cache[m], method=m, backend_name=bname)


if __name__ == "__main__":
    main()
