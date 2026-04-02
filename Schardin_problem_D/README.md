# Schardin Problem D

This folder contains a standalone Schardin Problem D solver built around a paper-style multispeed ELBM path rather than the old cylinder `D2Q9` core.

## What changed

- The lattice is now `D2Q49`, built from the `V7 = {0, +/-1, +/-2, +/-3}` one-dimensional admissible set.
- `Feq` is evaluated by numerical entropy minimization with a four-multiplier Newton solve.
- The collision step now uses an entropic line search instead of the old heuristic `alpha` switch.
- The wedge boundary reconstructs only populations that stream from solid into fluid-adjacent nodes, rather than overwriting the whole obstacle neighborhood.
- The left boundary is prescribed at the post-shock state, while the right/top/bottom boundaries reuse previous-step populations in the current implementation.
- The default YAML keeps the paper side-length resolution at `300` cells per wedge side.

## Benchmark assumptions

- Paper: `10.1103/PhysRevE.93.063302`
- Shock Mach number: `1.34`
- Reynolds number based on wedge side: `2000`
- Pre-shock state: `rho=1.4`, `p=1.0`, `u=v=0`
- Geometry: equilateral wedge with apex at `x/L = 1`, centered vertically at `y/L = 2.25`, apex facing upstream
- Domain: `4.0L x 4.5L`
- Shock: initialized at `x/L = 1`

The initialization follows open Schardin reproductions from the ELBM and DG literature for the same Mach-1.34 equilateral-triangle benchmark family.

## Run

From the repo root:

```bash
python Schardin_problem_D/schardin_solver.py --device 0 --scale 1.0 --steps 100 --plot_every 30
```

Outputs are written under `Schardin_problem_D/images/`.
