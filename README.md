# An Fourier approximation method — Project Structure

MATLAB code accompanying the paper *An Efficient Fourier Approximation Method for Maxwell’s Equations with Variable Coefficients*.

## Directory layout

```
An Fourier approximation method/
├── README.md                 # this file
├── example_2/                # Example 2: constant-coefficient Maxwell, epsilon = mu = 1
│   ├── ex2.m                 # Algorithm 4.1
│   ├── ex2_yee.m             # exact solution on the Yee grid (reference for ADI-FDTD)
│   └── ADIFDTD_3D.m          # classical ADI-FDTD solver
└── example_3/                # Example 3: variable-coefficient Maxwell (locally frozen coefficients)
    └── ex3_variable_coeff.m  # solver, error analysis, slice plots and convergence plot
```