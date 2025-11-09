# computations_pycuda.py
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

KERNEL = r"""
extern "C" {

__device__ __forceinline__
float fast_cosf(float x) { return cosf(x); }

__device__ __forceinline__
float fast_sinf(float x) { return sinf(x); }

__global__
void evaluate_field_kernel(
    const float* __restrict__ X,
    const float* __restrict__ u,
    float* __restrict__ out,
    const float V,
    const int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    float theta = X[3*i + 2];
    float dx = V * fast_cosf(theta);
    float dy = V * fast_sinf(theta);
    float dth = u[i];
    out[3*i + 0] = dx;
    out[3*i + 1] = dy;
    out[3*i + 2] = dth;
}

__global__
void euler_step_kernel(
    const float* __restrict__ X,
    const float* __restrict__ u,
    float* __restrict__ out,
    const float dt,
    const float V,
    const int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    float x  = X[3*i + 0];
    float y  = X[3*i + 1];
    float th = X[3*i + 2];

    float dx = V * fast_cosf(th);
    float dy = V * fast_sinf(th);
    float dth = u[i];

    out[3*i + 0] = x  + dt * dx;
    out[3*i + 1] = y  + dt * dy;
    out[3*i + 2] = th + dt * dth;
}

__global__
void euler_rollout_kernel(
    const float* __restrict__ X0,
    const float* __restrict__ u,
    float* __restrict__ out,
    const float dt,
    const float V,
    const int steps,
    const int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    float x  = X0[3*i + 0];
    float y  = X0[3*i + 1];
    float th = X0[3*i + 2];
    float ui = u[i];

    for (int s = 0; s < steps; ++s) {
        float dx  = V * fast_cosf(th);
        float dy  = V * fast_sinf(th);
        float dth = ui;
        x  += dt * dx;
        y  += dt * dy;
        th += dt * dth;
    }
    out[3*i + 0] = x;
    out[3*i + 1] = y;
    out[3*i + 2] = th;
}

}
"""

_mod = SourceModule(KERNEL, options=["-O3", "--use_fast_math"])
_evaluate_field = _mod.get_function("evaluate_field_kernel")
_euler_step = _mod.get_function("euler_step_kernel")
_euler_rollout = _mod.get_function("euler_rollout_kernel")


def to_backend(arr):
    a = np.asarray(arr, dtype=np.float32, order="C")
    return gpuarray.to_gpu(a)


def sync():
    drv.Context.synchronize()


def shape(x_gpu):
    return x_gpu.shape


def _grid1d(n, block=256):
    grid = (int((n + block - 1) // block), 1, 1)
    return (block, 1, 1), grid


def evaluate_field(X_gpu, u_gpu, V=1.0):
    N = int(X_gpu.shape[0])
    out = gpuarray.empty_like(X_gpu)
    block, grid = _grid1d(N)
    _evaluate_field(
        X_gpu, u_gpu, out,
        np.float32(V), np.int32(N),
        block=block, grid=grid)
    return out


def euler_step(X_gpu, u_gpu, dt, V=1.0):
    N = int(X_gpu.shape[0])
    out = gpuarray.empty_like(X_gpu)
    block, grid = _grid1d(N)
    _euler_step(
        X_gpu, u_gpu, out,
        np.float32(dt), np.float32(V), np.int32(N),
        block=block, grid=grid)
    return out


def euler_rollout(X0_gpu, u_gpu, dt, steps, V=1.0):
    N = int(X0_gpu.shape[0])
    out = gpuarray.empty_like(X0_gpu)
    block, grid = _grid1d(N)
    _euler_rollout(
        X0_gpu, u_gpu, out,
        np.float32(dt), np.float32(V), np.int32(steps), np.int32(N),
        block=block, grid=grid)
    return out
