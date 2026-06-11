# Composite-Expansion Physics-Informed Neural Networks (CE-PINN)

Code for **“Composite-expansion physics-informed neural networks for solving
boundary layer singularly perturbed problems.”**

CE-PINN combines physics-informed neural networks with the method of composite
expansions. Instead of training one network directly on a singularly perturbed
equation, it decomposes the solution into outer terms and boundary-layer
corrections. The resulting asymptotic subproblems are simpler and, for the main
cases, independent of the perturbation parameter.

This repository contains only the five CE-PINN examples used in the manuscript.
It intentionally excludes standard PINN, BL-PINN and C-PINN comparisons,
collocation sensitivity studies, plotting scripts, checkpoints, figures and
previously generated numerical results.

## Method

For a small perturbation parameter `epsilon`, CE-PINN represents a solution as

```text
u(x, epsilon) =
    sum_n delta_n(epsilon) u_outer_n(x)
  + sum_n delta_n(epsilon) u_boundary_n(xi),
```

where `xi = (x - x0) / delta(epsilon)` is a stretched coordinate near the
boundary layer. Separate PINN subnetworks approximate the outer and
boundary-layer terms.

The workflow has two stages:

1. **Offline training:** solve the hierarchy of asymptotic subproblems. Lower
   orders are trained first and then used as fixed source terms for higher
   orders.
2. **Online assembly:** substitute a desired small `epsilon` and combine the
   trained asymptotic terms. This is the “train once, solve for any small
   epsilon” part of CE-PINN.

Cases 1-3 train orders 0, 1 and 2 sequentially. Cases 4-5 retain the
nontrivial asymptotic terms derived for their coupled boundary-layer systems.

## Repository Layout

```text
CE-PINN/
├── examples/
│   ├── case1_1d_linear.py
│   ├── case2_1d_nonlinear.py
│   ├── case3_2d_linear.py
│   ├── case4_flat_plate.py
│   └── case5_unsteady_compressible.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Generated data are written to `outputs/` by default. That directory and common
checkpoint, NumPy and figure formats are excluded by `.gitignore`.

## Installation

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended
for full training, especially for the two-dimensional cases.

```bash
git clone https://github.com/<your-account>/CE-PINN.git
cd CE-PINN

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts automatically use CUDA when PyTorch detects an available GPU and
otherwise run on CPU.

## Common Training Settings

The defaults follow the revised manuscript:

| Setting | Default |
|---|---:|
| Hidden layers per subnetwork | 4 |
| Neurons per hidden layer | 64 |
| Activation | Swish |
| Optimizer | Adam |
| Initial learning rate | `1e-4` |
| Final cosine-schedule rate | `1e-6` |
| Training epochs | 10,000 |
| 1D collocation points | 100 |
| 2D points per coordinate | 50 |
| 2D batch size | 128 |
| Random seed | 1234 |

All scripts accept `--epochs`, `--npt`, `--batch-size`, `--seed` and
`--output-dir`. Use `python examples/<script>.py --help` for the complete
interface.

For a quick installation check, use one epoch and a small point set:

```bash
python examples/case1_1d_linear.py --epochs 1 --npt 4 --batch-size 4
```

This is only a smoke test and does not produce scientifically meaningful
accuracy.

## Case 1: 1D Linear Problem

The singularly perturbed convection-diffusion-reaction equation is

```text
epsilon u_xx + u_x + u = 0,       x in [0, 1],
u(0) = 0,  u(1) = 1.
```

The boundary layer is located at `x = 0`, has thickness `O(epsilon)`, and uses
`xi = x / epsilon`. The truncated stretched domain is `xi in [0, 20]`.

The script trains outer and boundary-layer subnetworks from order 0 through
order 2:

```bash
python examples/case1_1d_linear.py
```

To stop at a lower order:

```bash
python examples/case1_1d_linear.py --max-order 1
```

Outputs are stored in `outputs/case1_1d_linear/{0,1,2}/`. Each higher-order run
loads the previous order's best checkpoint automatically.

## Case 2: 1D Nonlinear Problem

The nonlinear benchmark is

```text
epsilon u_xx + u_x - u^2 = -1,    x in [0, 1],
u(0) = 0,  u(1) = 0.
```

Its boundary layer is also at `x = 0` with thickness `O(epsilon)`. The
stretched coordinate is `xi = x / epsilon`, truncated at `xi_max = 15`.

```bash
python examples/case2_1d_nonlinear.py
```

Orders 0, 1 and 2 are trained automatically and stored under
`outputs/case2_1d_nonlinear/`.

## Case 3: 2D Linear Elliptic Problem

The two-dimensional benchmark is

```text
epsilon (u_xx + u_yy) + u_x = -sin(pi y),
(x, y) in [0, 1] x [0, 1],
u = 0 on all four boundaries.
```

Only the `x` coordinate is stretched because the boundary layer is located at
`x = 0`: `xi = x / epsilon`, with `xi_max = 10`.

```bash
python examples/case3_2d_linear.py
```

The script trains orders 0-2 sequentially. The second-order residual implements
the equations given in Appendix A.3 of the revised manuscript, including the
two-dimensional inputs required by the outer and boundary-layer subnetworks.

## Case 4: Semi-Infinite Flat-Plate Boundary Layer

This case considers the normalized steady incompressible Navier-Stokes system
over a semi-infinite flat plate:

```text
u_x + v_y = 0,
u u_x + v u_y = -p_x + epsilon (u_xx + u_yy),
u v_x + v v_y = -p_y + epsilon (v_xx + v_yy).
```

The domain is `[0.25, 1] x [0, 1]`, excluding the leading-edge singularity.
The default Reynolds number is `1e6`, so `epsilon = 1e-6`. The wall-normal
stretched coordinate is `xi = y / sqrt(epsilon)` and `xi_max = 10`.

The retained CE-PINN terms follow the alternating outer/inner solution strategy
derived in the manuscript. The Blasius solution is evaluated internally for
boundary/reference data.

```bash
python examples/case4_flat_plate.py
```

The default output directory is `outputs/case4_flat_plate/`.

## Case 5: Unsteady Compressible Boundary Layer

This example models the impulsive motion of an infinite plate in a viscous
compressible fluid. In the density-weighted wall-normal coordinate, the
retained nontrivial inner equations are

```text
ub0_T - ub0_xixi = 0,
vb1_xi - hb0_T = 0,
hb0_T - hb0_xixi - (gamma - 1) Ma^2 (ub0_xi)^2 = 0.
```

The script trains only the terms requested for the `ucbl_b1` model:

- `ub0(xi, T)`: leading streamwise velocity correction;
- `hb0(xi, T)`: leading enthalpy correction;
- `vb1(xi, T)`: first normal-velocity correction.

The first-order outer acoustic fields are not included. The defaults are
`gamma = 1.4`, `Ma = 100/340`, `Re = 1e6`, `T in [0.25, 1]` and
`xi in [0, 10]`. The lower time boundary uses Van Dyke's large-time
asymptotic reference profiles.

```bash
python examples/case5_unsteady_compressible.py
```

To run another Reynolds number:

```bash
python examples/case5_unsteady_compressible.py --reynolds 1e5
```

The saved composite variables are

```text
u = ub0,
h = 1 + hb0,
rho = 1 / h,
v = sqrt(epsilon) vb1.
```

## Output Files

Depending on the case, a run creates:

| File | Description |
|---|---|
| `ckpt.pt` | Best network state and essential physical/training metadata |
| `loss_hist.npy` | Training loss history |
| `result_data.npz` | Evaluated fields and reference data when produced |

These files are reproducible artifacts and are deliberately not tracked by Git.

## Reproducibility Notes

- The default seed initializes NumPy and PyTorch.
- GPU and CPU runs may still differ slightly because of floating-point and
  backend behavior.
- Full 10,000-epoch training can be expensive. Do not judge accuracy from the
  one-epoch smoke-test command.
- Cases 1-3 must preserve order dependencies. Deleting an order-0 checkpoint
  before training order 1, or an order-1 checkpoint before order 2, will cause
  the run to fail.
- The repository contains training code only. Manuscript figures require
  separate post-processing and are intentionally excluded.

## Citation

If this code contributes to published work, cite the accompanying manuscript:

```text
Haonan Liu, Lei Zhang, Zhaobin Li, Composite-expansion physics-informed neural networks for solving boundary
layer singularly perturbed problems, revised manuscript under review (2026.6), Journal of Computational Physics
```

Replace this provisional citation with the final author list, volume, pages and
DOI after publication.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
