# 2D/3D Car‑like Dynamics — Pixel/Control Methods with CPU, Torch, and PyCUDA Backends

This repo contains **two related modules** that simulate a simple car‑like dynamical system in (x, y, θ) using several numerical integrators and compute/visualize reachable sets:

* **3D module** — grid‑based ("pixel") method on a 3D lattice over (x, y, θ). Includes visualization and a ready benchmark across backends (CPU / Torch / PyCUDA) and integrators (Euler, RK2, RK4).
* **2D module** — particle/cloud method that rolls out **N** independent states in parallel and benchmarks the same integrators across backends.

Both modules use the same kinematic model and share a consistent interface across backends to compare performance.

---

## 🔧 Mathematical model

State (X = (x, y, \theta)), control (u) (turn rate), constant forward speed (V):

[\dot x = V\cos\theta,\qquad \dot y = V\sin\theta,\qquad \dot\theta = u.]

Backends implement three explicit integrators:

* **Euler**
* **Runge–Kutta 2 (midpoint)**
* **Runge–Kutta 4 (classical)**

Backends:

* **CPU (NumPy)** — portable baseline
* **Torch** — CPU or CUDA, auto‑selected; can be forced to CPU
* **PyCUDA** — custom CUDA kernels

---

## 📦 Project layout (files & roles)

### 3D module (grid‑based / pixel method)

* **`pixel_method_3d.py`** — main entry point for the 3D pixel method:

  * Builds 3D grid over `(x, y, z)` where `z ≡ θ`.
  * Generates the set of admissible controls `all_u`.
  * Calls backend‐agnostic solvers to propagate the **front** of reachable voxels over time.
  * **Benchmarks** Euler/RK2/RK4 on CPU, Torch, PyCUDA and draws a **bar chart** comparing runtimes.
  * Visualization helpers: `draw2d(q, ...)` (heatmap slices) and `draw(m, ...)` (3D scatter of reachable set `m`).

* **`control_method_3d.py`** — alternative 3D visualizer for control method (static frames); chooses Torch vs CPU automatically by `torch.cuda.is_available()`.

* **`control_method2_3d.py`** — same idea as above with slightly different styling/flow for the rendering of multiple starting points / fronts.

* **`control_method_3d_gif.py`** — animation script using `matplotlib.animation` to produce a GIF of the front evolution over time.

* **`computations_cpu.py`** — NumPy backend for grid propagation & integrators.

  * Key functions (names may vary slightly by revision): `get_points`, `foo` (RHS), `euler`, `runge_kutta_2`, `runge_kutta_4`, `ij_to_xy`, `xy_to_ij`, `solve_control`, `solve`, `get_all_u`, `solve_pixel`.
  * Returns the reachable mask `q`, the set of visited indices `m`, the newly added shell `sm`, and timing.

* **`computations_torch.py`** — Torch backend (uses GPU if available). Same API as CPU version for comparability.

* **`computations_pycuda.py`** — PyCUDA backend with custom kernels. Same API contract; focuses on throughput of grid expansion.

* **`SCHEMA.md`** — reference doc explaining how `(i, j, k)` indices map to real coordinates and how `ij_to_xy` / `xy_to_ij` work, plus how the **front** propagates over the grid.

* **`Differential_Equasions_3D_Bench.ipynb`** — an exploratory notebook for timing/plots of the 3D setup.

### 2D module (particle/cloud method)

* **`control_main.py`** — CLI benchmark runner for 2D (actually still `(x, y, θ)` but in a particle model):

  * Spawns **N** independent particles with random initial states and per‑particle control `u`.
  * Benchmarks **Euler / RK2 / RK4** on available backends.
  * Produces a **bar plot** of runtime by backend & method.
  * **CLI options:**

    * `--n INT` (default `200_000`) — number of particles
    * `--steps INT` (default `200`) — number of integration steps
    * `--dt FLOAT` (default `1e-2`) — step size
    * `--V FLOAT` (default `1.0`) — forward speed
    * `--seed INT` (default `42`)
    * `--force_torch_cpu` — set `TORCH_FORCE_CPU=1` before import
    * `--disable_jit` — set `PYTORCH_JIT=0` before import

* **`pixel_method_2d_base.py`** — simple NumPy baseline of the same dynamics and integrators; helpful for correctness checks and small demos.

* **`computations_cpu.py`** — NumPy backend for particle rollouts:

  * API: `to_backend`, `sync`, `shape`, `evaluate_field`, `euler_step`, `euler_rollout`.

* **`computations_torch.py`** — Torch backend:

  * Device management helpers: `_set_device`, `force_cpu`, `backend_info`.
  * API: `to_backend`, `sync`, `shape`, `evaluate_field`, `euler_step`, `euler_rollout`.

* **`computations_pycuda.py`** — PyCUDA backend with kernels for field evaluation and Euler step; includes a simple 1D grid helper `_grid1d`.

> **Inter‑module relation.** Both modules implement the **same kinematics** and similar integrator APIs so you can compare algorithmic vs hardware effects in either a **grid‑front** (3D) or **particle** (2D) formulation.

---

## 🖥️ Requirements & installation

* Python 3.10+
* Packages (install what you need for your chosen backend):

  ```bash
  pip install numpy matplotlib seaborn
  # Optional backends
  pip install torch           # choose CUDA build from pytorch.org if using GPU
  pip install pycuda          # requires a working NVIDIA CUDA toolkit/driver
  ```
* **CUDA (optional):**

  * For Torch: install the PyTorch wheel matching your CUDA runtime & driver.
  * For PyCUDA: ensure `nvcc` and a compatible driver are installed.

Headless servers: set `matplotlib` backend or save figures to files instead of `plt.show()`.

---

## ▶️ Quick start

### 3D (grid/pixel method + benchmark)

Run and get both visuals and a runtime comparison:

```bash
python pixel_method_3d.py
```

This will:

1. Build a grid (defaults like `x∈[-5,5]`, `y∈[-5,5]`, `θ∈[-10,10]` with given steps).
2. Generate admissible controls `all_u` (elliptic constraint typical form `u1^2 + 25 u2^2 ≤ 1`).
3. Propagate the front over `t ∈ [0, 3π/2]` with step `dt` using **Euler, RK2, RK4** on **CPU/Torch/PyCUDA**.
4. Render 3D reachable set scatter and 2D heatmaps.
5. Plot a **bar chart** of total times per backend & method.

For static frames/alternative views:

```bash
python control_method_3d.py
python control_method2_3d.py
```

For an animation (GIF):

```bash
python control_method_3d_gif.py
```

### 2D (particle/cloud method + benchmark)

Run the large‑N rollout benchmark and produce a bar chart:

```bash
python control_main.py --n 200000 --steps 200 --dt 1e-2 --V 1.0
```

Flags to control Torch behavior:

```bash
python control_main.py --force_torch_cpu        # ignore CUDA even if present
python control_main.py --disable_jit            # disable TorchScript/JIT
```

Expected output:

* Console: per‑backend initialization info and per‑method elapsed times.
* Figure: grouped bar chart **CPU vs TORCH vs PYCUDA** for **Euler / RK2 / RK4**.

> Tip: increase `--n` and `--steps` for stress tests; keep an eye on GPU memory for Torch/PyCUDA.

---

## 🧪 Benchmark methodology

Both modules benchmark the same **numerical kernels** under different compute backends.

**What’s measured:** wall‑clock time to complete a fixed number of integration steps:

* **2D:** `N` particles rolled out for `steps` steps with step `dt`.
* **3D:** number of time layers over `t` with fixed grid resolution `(nx, ny, nz)` and admissible controls set size.

**Integrators:** Euler, RK2, RK4.

**Backends:** CPU (NumPy), Torch (CPU or CUDA), PyCUDA.

**Outputs:** a dict of times per (method, backend) and a bar plot summarizing results. The 3D script also displays intermediate reachable‐set plots for sanity checks.

**Reproducibility:**

* 2D: `--seed` controls random initial conditions and controls.
* 3D: grid ranges/steps and `t` grid are deterministic given the script defaults.
* Torch GPU runs call `torch.cuda.synchronize()` where appropriate; PyCUDA kernels synchronize before timing is recorded.

---

## ⚙️ Configuration knobs

* **Grid (3D):** `x_start/x_finish/nx`, `y_start/...`, `z_start/...`; time grid `t_start/t_finish/nt` with `dt = t[1]-t[0]`.
* **Particle count (2D):** `--n`, `--steps`, `--dt`, `--V`.
* **Controls:** 3D `get_all_u(...)` builds an admissible set; adjust to change the turning‑rate envelope.
* **Backends:** 3D scripts auto‑import Torch vs CPU by availability; 2D CLI runs all present backends and labels missing ones.

---

## 🖼️ Visualization

* **3D**

  * `draw(m, title)` — scatter of reachable voxels
  * `draw2d(q, title)` — 2D heatmap slices with labeled ticks
  * `control_method_3d_gif.py` — time‑lapse animation
* **2D**

  * `control_main.py` generates the benchmark **bar chart**

Use `plt.savefig("file.png", dpi=200, bbox_inches="tight")` if running headless.

---

## 🩺 Troubleshooting

* **Torch: `CUDA driver error: invalid resource handle`**

  * Usually indicates a stale CUDA context. Restart the Python process/kernel.
  * Ensure driver/toolkit compatibility; try `--force_torch_cpu` as a fallback.

* **PyCUDA: `TypeError: invalid type on parameter #0 (0-based)`**

  * Occurs if a kernel expects GPU pointers but receives plain Python/NumPy scalars.
  * Use the provided wrappers (`to_backend`, `evaluate_field`, `euler_step`) that manage `gpuarray` types.
  * Ensure arrays are `float32` and contiguous.

* **Matplotlib windows don’t open**

  * Use a non‑interactive backend or save plots to files.

---

## 📚 Notes

* See **`SCHEMA.md`** in the 3D module for the precise definition of index ↔ coordinate transforms and the "front" expansion logic.
* Both modules are designed for side‑by‑side **algorithmic** and **hardware** comparisons. Keep the APIs aligned when extending functionality to maintain a fair benchmark.

---

## 📝 License & authorship

Add your preferred license here. Replace this section with authors/affiliations if needed.
