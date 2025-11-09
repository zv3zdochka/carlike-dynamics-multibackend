# computations_cuda.py
from time import perf_counter
import numpy as np

V_ = 1.0
L_ = 1.0
PHI_MAX = np.radians(45)
LINE_INTERPOLATION = False

_pycuda_available = False
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    _pycuda_available = True
except Exception:
    _pycuda_available = False

device = 'cuda' if _pycuda_available else 'cpu'


def get_points(ij_st, ij_fn):
    if LINE_INTERPOLATION:
        st = np.asarray(ij_st)
        fn = np.asarray(ij_fn)
        n = int(np.max(np.abs(st - fn)))
        if n < 1:
            return fn.reshape(-1, 1)
        iijj = np.linspace(st, fn, n + 1, dtype=np.int32).T
        return iijj
    return np.asarray(ij_fn).reshape(-1, 1)


def _foo_cpu(_, x_, u_):
    z = x_[2]
    cos_z = np.cos(z) * V_
    sin_z = np.sin(z) * V_
    zeros = np.zeros_like(z, dtype=np.float32) + u_
    return np.stack([cos_z, sin_z, zeros], axis=0)


def euler(t0, dt_, x_, u_, func):
    deriv = func(t0, x_, u_)
    return x_ + deriv * dt_


def runge_kutta_2(t0, dt_, x_, u_, func):
    dt_val = float(dt_)
    half_dt = dt_val / 2.0
    mid = _foo_cpu(t0 + half_dt, x_, u_)
    x_mid = x_ + mid * half_dt
    k = func(t0, x_mid, u_)
    return x_ + k * dt_val


def runge_kutta_4(t0, dt_, x_, u_, func):
    dt_val = float(dt_)
    half_dt = dt_val / 2.0
    k1 = _foo_cpu(t0, x_, u_)
    k2 = _foo_cpu(t0 + half_dt, x_ + k1 * half_dt, u_)
    k3 = _foo_cpu(t0 + half_dt, x_ + k2 * half_dt, u_)
    k4 = _foo_cpu(t0 + dt_val, x_ + k3 * dt_val, u_)
    return x_ + (k1 + 2 * k2 + 2 * k3 + k4) * (dt_val / 6.0)


def ij_to_xy(ij, x_start, dx, y_start, dy, z_start, dz):
    arr = np.asarray(ij, dtype=np.float32)
    xs = arr[0] * dx + x_start
    ys = arr[1] * dy + y_start
    zs = arr[2] * dz + z_start
    return np.stack([xs, ys, zs], axis=0)


def xy_to_ij(xy, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz):
    arr = np.asarray(xy)
    ix = np.clip(np.round((arr[0] - x_start) / dx), 0, nx - 1).astype(np.int32)
    iy = np.clip(np.round((arr[1] - y_start) / dy), 0, ny - 1).astype(np.int32)
    iz = np.clip(np.round((arr[2] - z_start) / dz), 0, nz - 1).astype(np.int32)
    return np.stack([ix, iy, iz], axis=0)


def solve_control(start_point, control, method='RK4', t=None, dt=None,
                  x_start=None, dx=None, nx=None,
                  y_start=None, dy=None, ny=None,
                  z_start=None, dz=None, nz=None):
    start_point_np = np.asarray(start_point, dtype=np.float32)
    t_np = np.asarray(t, dtype=np.float32)
    dt_val = float(dt)
    res = set()
    for i_idx in range(len(t_np)):
        for j_idx in range(i_idx, len(t_np)):
            xx_new = start_point_np.copy()
            for k_idx in range(len(t_np)):
                if k_idx <= i_idx:
                    u_k = control[0]
                elif k_idx <= j_idx:
                    u_k = control[1]
                else:
                    u_k = control[2]
                if method == 'RK4':
                    xx_new = runge_kutta_4(t_np[k_idx], dt_val, xx_new, u_k, _foo_cpu)
                elif method == 'RK2':
                    xx_new = runge_kutta_2(t_np[k_idx], dt_val, xx_new, u_k, _foo_cpu)
                elif method == 'Euler':
                    xx_new = euler(t_np[k_idx], dt_val, xx_new, u_k, _foo_cpu)
                else:
                    raise NotImplementedError(f"Unknown method: {method}")
            ij_new = xy_to_ij(xx_new, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
            if ij_new.ndim == 1:
                ij_new = ij_new.reshape(3, 1)
            res.add((int(ij_new[0, 0]), int(ij_new[1, 0]), int(ij_new[2, 0])))
    return res


def solve(start_point, controls_, method='RK4', t=None, dt=None,
          x_start=None, dx=None, nx=None,
          y_start=None, dy=None, ny=None,
          z_start=None, dz=None, nz=None):
    time_start = perf_counter()
    list_of_pts = []
    for control in controls_:
        points = solve_control(start_point, control=control, method=method,
                               t=t, dt=dt,
                               x_start=x_start, dx=dx, nx=nx,
                               y_start=y_start, dy=dy, ny=ny,
                               z_start=z_start, dz=dz, nz=nz)
        list_of_pts.append(points)
    time_spent = perf_counter() - time_start
    print(f"{method} Time: {time_spent}")
    return list_of_pts


def get_all_u(u_):
    return np.asarray([float(u1) for u1 in u_], dtype=np.float32)


_GPU_SRC = r"""
// fast-math
extern "C" __global__
void integrate_and_mark(
    
    const int* __restrict__ sm_i,
    const int* __restrict__ sm_j,
    const int* __restrict__ sm_k,
    const int M,               

    
    const float* __restrict__ u_vals,
    const int Nu,             

    
    const float x_start, const float dx, const int nx,
    const float y_start, const float dy, const int ny,
    const float z_start, const float dz, const int nz,
    const float dt,            
    const int method_id,       // 0=Euler,1=RK2,2=RK4
    const float V,             

    // visited
    int* __restrict__ visited,       // flatten [nx*ny*nz]
    const int* __restrict__ visited_prev, 

    int* __restrict__ out_i,
    int* __restrict__ out_j,
    int* __restrict__ out_k,
    int* __restrict__ out_count      
){
    const int R = M * Nu;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;

    const int p = r % M;     
    const int c = r / M;     

    // base ijk -> xyz (phi в z)
    const int ii = sm_i[p];
    const int jj = sm_j[p];
    const int kk = sm_k[p];

    float x = x_start + (float)ii * dx;
    float y = y_start + (float)jj * dy;
    float phi = z_start + (float)kk * dz;

    const float u = u_vals[c];
    const float half_dt = 0.5f * dt;

    // f(x) = [V*cos(phi), V*sin(phi), u]
    auto f_x = [&](float phi_val){ return V * __cosf(phi_val); };
    auto f_y = [&](float phi_val){ return V * __sinf(phi_val); };

    float xn, yn, phin;

    if (method_id == 0){
        // Euler
        xn   = x   + f_x(phi) * dt;
        yn   = y   + f_y(phi) * dt;
        phin = phi + u * dt;
    } else if (method_id == 1){
        // RK2 (midpoint)
        float k1x = f_x(phi);
        float k1y = f_y(phi);
        float k1p = u;

        float xm = x   + k1x * half_dt;
        float ym = y   + k1y * half_dt;
        float pm = phi + k1p * half_dt;

        float k2x = f_x(pm);
        float k2y = f_y(pm);
        float k2p = u;

        xn   = x   + k2x * dt;
        yn   = y   + k2y * dt;
        phin = phi + k2p * dt;
    } else {
        // RK4
        float k1x = f_x(phi);
        float k1y = f_y(phi);
        float k1p = u;

        float k2x = f_x(phi + k1p * half_dt);
        float k2y = f_y(phi + k1p * half_dt);
        float k2p = u;

        float k3x = f_x(phi + k2p * half_dt);
        float k3y = f_y(phi + k2p * half_dt);
        float k3p = u;

        float k4x = f_x(phi + k3p * dt);
        float k4y = f_y(phi + k3p * dt);
        float k4p = u;

        xn   = x   + (k1x + 2.f*k2x + 2.f*k3x + k4x) * (dt/6.f);
        yn   = y   + (k1y + 2.f*k2y + 2.f*k3y + k4y) * (dt/6.f);
        phin = phi + (k1p + 2.f*k2p + 2.f*k3p + k4p) * (dt/6.f);
    }

    // map xy(phi) -> ijk
    int i2 = (int)rintf((xn - x_start) / dx);
    int j2 = (int)rintf((yn - y_start) / dy);
    int k2 = (int)rintf((phin - z_start) / dz);

    // clamp
    if (i2 < 0) i2 = 0; else if (i2 >= nx) i2 = nx - 1;
    if (j2 < 0) j2 = 0; else if (j2 >= ny) j2 = ny - 1;
    if (k2 < 0) k2 = 0; else if (k2 >= nz) k2 = nz - 1;

    // flatten index
    const int idx = (i2 * ny + j2) * nz + k2;

    int was = atomicExch(&visited[idx], 1);
    if (was == 0 && visited_prev[idx] == 0){
        int pos = atomicAdd(out_count, 1);
        out_i[pos] = i2;
        out_j[pos] = j2;
        out_k[pos] = k2;
    }
}
"""


def _compile_module():
    return SourceModule(_GPU_SRC, options=['-use_fast_math'])


def solve_pixel(method='RK4', t=None, dt=None, all_u=None, start_p=None,
                x_start=None, dx=None, nx=None,
                y_start=None, dy=None, ny=None,
                z_start=None, dz=None, nz=None):
    time_start = perf_counter()

    start_p_np = np.asarray(start_p, dtype=np.float32)
    t_np = np.asarray(t, dtype=np.float32)
    dt_val = float(dt)
    all_u_np = np.asarray(all_u, dtype=np.float32)

    if _pycuda_available:
        mod = _compile_module()
        kern = mod.get_function('integrate_and_mark')

        n_cells = int(nx * ny * nz)

        # visited: int32 (0/1), flatten
        visited = gpuarray.zeros(n_cells, dtype=np.int32)
        visited_prev = gpuarray.zeros(n_cells, dtype=np.int32)

        idx0 = xy_to_ij(start_p_np, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
        i0, j0, k0 = int(idx0[0, 0]), int(idx0[1, 0]), int(idx0[2, 0])
        start_flat = (i0 * ny + j0) * nz + k0
        visited[start_flat] = 1

        sm_i = gpuarray.to_gpu(np.array([i0], dtype=np.int32))
        sm_j = gpuarray.to_gpu(np.array([j0], dtype=np.int32))
        sm_k = gpuarray.to_gpu(np.array([k0], dtype=np.int32))
        M = 1

        u_vals = gpuarray.to_gpu(all_u_np)

        out_i = None
        out_j = None
        out_k = None
        out_count = gpuarray.zeros(1, dtype=np.int32)

        method_id = {'Euler': 0, 'RK2': 1, 'RK4': 2}[method]

        for time_idx, current_time in enumerate(t_np):
            if M == 0:
                break

            visited_prev.set(visited)  # device-to-device copy

            R = int(M * all_u_np.size)
            if (out_i is None) or (out_i.size < R):
                out_i = gpuarray.empty(R, dtype=np.int32)
                out_j = gpuarray.empty(R, dtype=np.int32)
                out_k = gpuarray.empty(R, dtype=np.int32)
            out_count.fill(0)

            block = 256
            grid = (R + block - 1) // block

            kern(
                sm_i, sm_j, sm_k, np.int32(M),
                u_vals, np.int32(all_u_np.size),
                np.float32(x_start), np.float32(dx), np.int32(nx),
                np.float32(y_start), np.float32(dy), np.int32(ny),
                np.float32(z_start), np.float32(dz), np.int32(nz),
                np.float32(dt_val), np.int32(method_id), np.float32(V_),
                visited, visited_prev,
                out_i, out_j, out_k, out_count,
                block=(block, 1, 1), grid=(grid, 1, 1)
            )

            new_M = int(out_count.get()[0])
            print(float(current_time), new_M)

            if new_M == 0:
                M = 0
                sm_i = gpuarray.empty(0, dtype=np.int32)
                sm_j = gpuarray.empty(0, dtype=np.int32)
                sm_k = gpuarray.empty(0, dtype=np.int32)
            else:
                sm_i = out_i[:new_M]
                sm_j = out_j[:new_M]
                sm_k = out_k[:new_M]
                M = new_M

        visited_host = visited.get().reshape(nx, ny, nz).astype(np.float32)

        nz_idx = np.array(np.nonzero(visited_host), dtype=np.int32).T
        m = set(map(tuple, nz_idx.tolist()))

        if M > 0:
            last_idx = np.stack([sm_i.get(), sm_j.get(), sm_k.get()], axis=1)
            sm = set(map(tuple, last_idx.tolist()))
        else:
            sm = set()

        time_spent = perf_counter() - time_start
        print(method, 'Time: ', time_spent)
        return visited_host, m, sm, time_spent

    visited = np.zeros((nx, ny, nz), dtype=bool)
    idx = xy_to_ij(start_p_np, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
    visited[int(idx[0, 0]), int(idx[1, 0]), int(idx[2, 0])] = True
    sm_idx = np.array(np.nonzero(visited), dtype=np.int32)
    for time_idx, current_time in enumerate(t_np):
        if sm_idx.size == 0:
            break
        old_visited = visited.copy()
        xx_ = ij_to_xy(sm_idx, x_start, dx, y_start, dy, z_start, dz)
        num_points = xx_.shape[1]
        xx_new_all = np.zeros((3, num_points * len(all_u_np)), dtype=np.float32)
        for iu, u_val in enumerate(all_u_np):
            if method == 'RK4':
                xx_new = runge_kutta_4(current_time, dt_val, xx_, u_val, _foo_cpu)
                angle = xx_new[2]
                xx_new = np.stack([xx_new[0], xx_new[1], angle], axis=0)
            elif method == 'RK2':
                xx_new = runge_kutta_2(current_time, dt_val, xx_, u_val, _foo_cpu)
            elif method == 'Euler':
                xx_new = euler(current_time, dt_val, xx_, u_val, _foo_cpu)
            else:
                raise NotImplementedError(f"Unknown method: {method}")
            start = iu * num_points
            end = start + num_points
            xx_new_all[:, start:end] = xx_new
        ij_new_all = xy_to_ij(xx_new_all, x_start, dx, nx, y_start, dy, ny, z_start, dz, nz)
        if LINE_INTERPOLATION:
            m_new = set()
            for r in range(ij_new_all.shape[1]):
                line = get_points(sm_idx[:, r % num_points], ij_new_all[:, r])
                for i_idx in range(line.shape[1]):
                    m_new.add((int(line[0, i_idx]), int(line[1, i_idx]), int(line[2, i_idx])))
            for pt in m_new:
                visited[pt[0], pt[1], pt[2]] = True
        else:
            visited[ij_new_all[0, :], ij_new_all[1, :], ij_new_all[2, :]] = True
        new_mask = visited & ~old_visited
        sm_idx = np.array(np.nonzero(new_mask), dtype=np.int32)
        print(float(current_time), sm_idx.shape[1])
    q = visited.astype(np.float32)
    all_idx = np.array(np.nonzero(visited)).T
    m = set(map(tuple, all_idx.tolist()))
    last_idx = np.array(np.nonzero(new_mask)).T if 'new_mask' in locals() else np.empty((0, 3), dtype=np.int32)
    sm = set(map(tuple, last_idx.tolist()))
    time_spent = perf_counter() - time_start
    print(method, 'Time: ', time_spent)
    return q, m, sm, time_spent
