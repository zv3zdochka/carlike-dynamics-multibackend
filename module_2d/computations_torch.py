# computations_torch.py
import os
import torch

try:
    torch.jit._state.disable()
except Exception:
    pass

_dtype = torch.float32
_device = torch.device("cpu")


def _can_use_cuda_safely() -> bool:
    if os.getenv("TORCH_FORCE_CPU", "") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.ones(1, device="cuda", dtype=_dtype)
        t = t + 1
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _set_device():
    global _device
    _device = torch.device("cuda") if _can_use_cuda_safely() else torch.device("cpu")


_set_device()


def force_cpu():
    global _device
    _device = torch.device("cpu")


def backend_info():
    if _device.type == "cuda":
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"torch on CUDA:{idx} ({name})"
        except Exception:
            return "torch on CUDA (name unavailable)"
    return "torch on CPU"


def to_backend(arr):
    if isinstance(arr, torch.Tensor):
        if _device.type == "cpu" and arr.device.type == "cuda":
            return arr.detach().to(device=_device, dtype=_dtype)
        return arr.detach().to(device=_device, dtype=_dtype, non_blocking=False)
    return torch.as_tensor(arr, dtype=_dtype, device=_device)


def sync():
    if _device.type == "cuda":
        torch.cuda.synchronize()


def shape(x):
    return tuple(x.shape)


def _run_with_fallback(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e)
        if "CUDA" in msg or "cuda" in msg:
            force_cpu()
            new_args = []
            for a in args:
                if isinstance(a, torch.Tensor) and a.device.type == "cuda":
                    new_args.append(a.to("cpu"))
                else:
                    new_args.append(a)
            for k, v in list(kwargs.items()):
                if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                    kwargs[k] = v.to("cpu")
            return fn(*new_args, **kwargs)
        raise


def _evaluate_field_no_guard(X, u, V: float):
    theta = X[:, 2]
    dx = torch.cos(theta) * V
    dy = torch.sin(theta) * V
    dth = u
    out = torch.empty_like(X)
    out[:, 0] = dx
    out[:, 1] = dy
    out[:, 2] = dth
    return out


def evaluate_field(X, u, V: float = 1.0):
    with torch.no_grad():
        return _run_with_fallback(_evaluate_field_no_guard, X, u, float(V))


def _euler_step_no_guard(X, u, dt: float, V: float):
    theta = X[:, 2]
    dx = torch.cos(theta) * V
    dy = torch.sin(theta) * V
    dth = u
    return X + dt * torch.stack((dx, dy, dth), dim=1)


def euler_step(X, u, dt, V: float = 1.0):
    with torch.no_grad():
        return _run_with_fallback(_euler_step_no_guard, X, u, float(dt), float(V))


def _euler_rollout_no_guard(X0, u, dt: float, steps: int, V: float):
    X = X0.clone()
    for _ in range(int(steps)):
        theta = X[:, 2]
        dx = torch.cos(theta) * V
        dy = torch.sin(theta) * V
        dth = u
        X = X + dt * torch.stack((dx, dy, dth), dim=1)
    return X


def euler_rollout(X0, u, dt, steps: int, V: float = 1.0):
    with torch.no_grad():
        return _run_with_fallback(_euler_rollout_no_guard, X0, u, float(dt), int(steps), float(V))
