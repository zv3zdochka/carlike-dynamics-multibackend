# Car-like Dynamics Benchmark (2D & 3D)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Backends](https://img.shields.io/badge/Backends-NumPy%20%7C%20Torch%20%7C%20PyCUDA-green)
![Integrators](https://img.shields.io/badge/Integrators-Euler%2C%20RK2%2C%20RK4-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

This repository contains two complementary modules for simulating a car-like kinematic model and benchmarking explicit
time-stepping schemes across multiple compute backends. The modules share the same dynamics and aligned APIs so results
are directly comparable.

- **3D module (`module_3d/`)** — grid-based (“pixel”) propagation of the reachable set on a lattice over \((
  x,y,\theta)\); includes visualizations and a ready-to-run benchmark.
- **2D module (`module_2d/`)** — particle/cloud rollout of \(N\) independent states; focuses on throughput and backend
  comparisons.

Вот цельный рабочий Markdown-кусок, который корректно отображается на GitHub без ошибок форматирования:

---

## 1. Model

**State equations**

**State:** ![state](https://latex.codecogs.com/svg.image?\color{white}X=(x,y,\theta))  
**Control:** ![control](https://latex.codecogs.com/svg.image?\color{white}u\text{~(turn%20rate)},\;V\text{~–~constant%20forward%20speed})

**Equations of Motion:**

![eq](https://latex.codecogs.com/svg.image?\color{white}\begin{cases}\dot{x}=V\cos\theta,\\[4pt]\dot{y}=V\sin\theta,\\[4pt]\dot{\theta}=u.\end{cases})


**Available explicit integrators (in all backends):**
- Euler  
- Runge–Kutta 2 (midpoint)  
- Runge–Kutta 4 (classical)
Available explicit integrators (in all backends):

- Euler  
- Runge–Kutta 2 (midpoint)  
- Runge–Kutta 4 (classical)

Backends:

- **CPU (NumPy)** — portable baseline  
- **Torch** — CPU or CUDA (auto-selected; can be forced to CPU)  
- **PyCUDA** — custom CUDA kernels

---

## 2. Repository layout

```

module_2d/
**init**.py
computations_cpu.py         # NumPy backend
computations_torch.py       # Torch backend (CPU/CUDA)computations_pycuda.py      # PyCUDA backend
control_main.py             # CLI benchmark (plots bar chart)
pixel_method_2d_base.py     # minimal NumPy demo/baseline

module_3d/
**init**.py
computations_cpu.py         # NumPy backend (grid utilities: ij_to_xy, xy_to_ij, ...)
computations_torch.py       # Torch backend
computations_pycuda.py      # PyCUDA backend (grid expansion kernels)
pixel_method_3d.py          # main: grid build, front propagation, benchmark, plots
control_method_3d.py        # alt static visualization
control_method2_3d.py       # alt static visualization (variant)
control_method_3d_gif.py    # animation via matplotlib.animation
SCHEMA.md                   # index↔coordinate mapping & front propagation

Differential_Equasions_2D_Bench.ipynb   # exploratory timings/plots (2D)
Differential_Equasions_3D_Bench.ipynb   # exploratory timings/plots (3D)
requirements.txt
README.md

````

---

## 3. Installation

- Python 3.10+
- Install only these libraries:

```bash
pip install numpy matplotlib
pip install torch           # install CUDA-enabled wheel if using GPU
pip install pycuda          # requires NVIDIA driver + CUDA toolkit (nvcc)
````

Rr you can install all the previous libraries from Colab (9.11.2025).

```bash
pip install -r requirements.txt
```

**CUDA notes**

* Torch: use a wheel matching your driver/runtime.
* PyCUDA: ensure `nvcc` is on `PATH` and driver/toolkit versions are compatible.

Headless environments: set a non-interactive Matplotlib backend or save figures to files instead of calling
`plt.show()`.

---

## 4. Quick start

### 4.1 3D reachable set (grid/pixel) with benchmark

```bash
python module_3d/pixel_method_3d.py
```

This script:

1. builds a 3D grid (defaults: (x,y\in[-5,5]), (\theta\in[-10,10]));
2. generates admissible controls (e.g., an elliptic set (u_1^2 + 25u_2^2 \le 1));
3. propagates the front over (t\in[0, 3\pi/2]) with step (dt) using Euler/RK2/RK4 on CPU/Torch/PyCUDA;
4. renders a 3D scatter of the reachable set and 2D heatmap slices;
5. produces a runtime bar chart by backend/integrator.

Alternative views:

```bash
python module_3d/control_method_3d.py
python module_3d/control_method2_3d.py
```

Animation:

```bash
python module_3d/control_method_3d_gif.py
```

### 4.2 2D particle/cloud rollout with benchmark

```bash
python module_2d/control_main.py --n 200000 --steps 200 --dt 1e-2 --V 1.0
```

Torch control flags:

```bash
python module_2d/control_main.py --force_torch_cpu   # ignore CUDA even if present
python module_2d/control_main.py --disable_jit       # disable TorchScript/JIT
```

**Outputs**

* Console: per-backend initialization info and elapsed times per integrator.
* Figure: grouped bar chart (CPU vs Torch vs PyCUDA; Euler/RK2/RK4).

---

## 5. Benchmark methodology

**What is measured** — wall-clock time for a fixed amount of work:

* **2D:** (N) particles × `steps` with step `dt`.
* **3D:** fixed grid ((n_x,n_y,n_z)), fixed number of time layers, fixed control set size.

**Integrators** — Euler, RK2, RK4.
**Backends** — NumPy (CPU), Torch (CPU/CUDA), PyCUDA (CUDA).
**Results** — timings are printed to stdout; the driver scripts also render bar charts.

**Reproducibility**

* 2D: use `--seed` to fix random initial states/controls.
* 3D: the grid/time grids are deterministic given the defaults.
* GPU runs call explicit synchronizations (Torch `cuda.synchronize()` / PyCUDA stream sync) before timing.

> Tip: for stress tests, increase `--n` / `--steps` (2D) or grid resolution/time layers (3D). Monitor GPU memory
> pressure when using Torch/PyCUDA.

---

## 6. Configuration

* **3D grid** — `x_start/x_finish/nx`, `y_start/...`, `z_start/...`; time grid `t_start/t_finish/nt` with
  `dt = t[1]-t[0]`.
* **2D particles** — `--n`, `--steps`, `--dt`, `--V`.
* **Controls (3D)** — built via `get_all_u(...)`; tune to change the turning-rate envelope.
* **Backends** — 3D scripts auto-select Torch vs CPU; the 2D CLI runs all available backends and reports missing ones.

---

## 7. Visualization

* **3D**

    * `draw(m, title)` — 3D scatter of reachable voxels.
    * `draw2d(q, title)` — heatmap slices (indexed grid).
    * `control_method_3d_gif.py` — time-lapse animation.
* **2D**

    * `control_main.py` — grouped bar chart of runtimes.

When running headless:

```python
plt.savefig("figure.png", dpi=200, bbox_inches="tight")
```




