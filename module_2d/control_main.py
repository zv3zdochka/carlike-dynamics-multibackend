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
from pathlib import Path

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


if __name__ == "__main__":
    main()
