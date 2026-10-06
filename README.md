# An Fourier approximation method — Project Structure

MATLAB code accompanying the paper *A Fourier approximation method for the 3D time-dependent Maxwell equations*. The scripts below correspond to the numerical examples and tables/figures in the paper.

## Directory layout

```
An Fourier approximation method/
├── README.md                 # this file
├── example_2/                # Example 2: constant-coefficient Maxwell, epsilon = mu = 1
│   ├── ex2.m                 # Algorithm 4.1
│   ├── ex2_yee.m             # exact solution on the Yee grid (reference for ADI-FDTD)
│   ├── ADIFDTD_3D.m          # classical ADI-FDTD solver
│   └── ex2_exact_HEn_*.mat   # pre-computed exact-solution data (M = 50, N = 8/16/32/64/128, t = 0.1/1.0)
└── example_3/                # Example 3: variable-coefficient Maxwell (locally frozen coefficients)
    └── ex3_variable_coeff.m  # solver, error analysis, slice plots and convergence plot
```

## Role of each script in the paper

| Script | Role |
| --- | --- |
| `example_2/ex2.m` | Implementation of **Algorithm 4.1**; produces **Tables 2, 3, 4, 5** and **Figure 1**. |
| `example_2/ex2_yee.m` | Computes the exact solution on the Yee grid and saves it as `.mat` files, used as the reference to validate the **ADI-FDTD** method. The required exact-solution data have already been generated and committed, so this script does not need to be run. |
| `example_2/ADIFDTD_3D.m` | Classical ADI-FDTD solver; produces **Table 6** and **Table 7**. It reads the pre-computed exact-solution `.mat` files from `example_2/`. |
| `example_3/ex3_variable_coeff.m` | Solves the variable-coefficient problem with `mu = 2 + sin(...)`, `eps = 2 + cos(...)`. Uses the locally frozen-coefficient method, computes errors at `t = 1.0 s` and `t = 50.0 s` on a `16^3` grid for `M ∈ {2,4,8,16}` (reference `M = 50`), prints LaTeX tables, produces 2D slice/error figures at `x3 = 0.5` (`N_slice = 64`) and saves a semi-log convergence plot. |