## NACA0012 Aerofoil

Scaffold for the `Supersonic flow past NACA0012 aerofoil` case referenced by:

`Tran, Leong, Le, Le, Computers & Fluids 249 (2022) 105701`
`DOI: 10.1016/j.compfluid.2022.105701`

What is now stored from the user-supplied paper benchmark excerpt:

- D2Q9 two-population thermal LBM
- transonic: `Ma∞ = 0.8, 0.9`, `Re = 500`, `alpha = 0`, `U = 0.2u_inf`, domain `20C x 20C`, head at `6.75C`, `C = 100`
- supersonic: `Ma∞ = 1.5`, `Re = 1e4 .. 1e7`, `alpha = 0`, `U = 0.4u_inf`, domain `7C x 4C`, head at `2C/3`, `C = 150` and `300`
- BC family: left Dirichlet on velocity/temperature/density, right Neumann on velocity/temperature/density, top/bottom freestream, aerofoil wall no-slip with no-penetration for temperature and density
- `Pr = 0.71`
- `gamma = 1.4`

What is not verified locally yet:

- freestream state variables beyond the high-level nondimensional numbers
- plotting targets used in the paper benchmark section
- exact finite-volume / lattice update formulas beyond the benchmark summary

Correction:

- The earlier `X = 30001, Y = 5` carry-over was not from the paper. It came from a copied repo utility example and was wrong for this directory.
- That misleading synthetic-data setup has been removed from the case config and disabled in the utility file.

Current status:

- `airfoil_param.yml` now contains the benchmark scenarios, BC family, and the stabilization notes from the paper summary you supplied.
- `airfoil_solver.py`, `train_stage_1.py`, `train_stage_2.py`, and `eval.py` are honest scaffolds rather than a false cylinder-based implementation.
- `plot_naca0012_geometry.py` now renders a setup schematic for both transonic and supersonic paper scenarios.
