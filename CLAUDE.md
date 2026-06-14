# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A multiphysics dynamics simulation package for GMAW/MIG arc welding, written in Python (NumPy/SciPy/Matplotlib). It models the full chain from arc circuit → droplet transfer → workpiece heat conduction, plus a variational-integrator extension for oscillation/contact problems. There is no web/service layer — it is a pure simulation library with two CLI entry points that produce PNG plots.

Note: the README and most code comments/docstrings are in Chinese. Match that convention when editing existing modules.

## Commands

```bash
uv sync             # create .venv and install deps (numpy, scipy, matplotlib)
uv run welding-sim      # run modules 1–5, writes plots to ./results/
uv run welding-sim-vi   # run variational modules 6–8, writes plots to ./results/
```

There is **no test suite, linter, or formatter configured.** Validation is done by running the two CLIs and inspecting the printed steady-state numbers (compared against the reference table in README.md) and the regenerated `results/*.png`. When changing physics, run the relevant CLI and confirm the printed values still match the README's "典型结果" table.

To exercise a single module without running everything, import the class directly:

```python
uv run python -c "from welding_dynamics import GMAWDynamics; print(GMAWDynamics().simulate()['I'][-1])"
```

## Architecture

The package (`src/welding_dynamics/`) is a set of independent physics modules, each a self-contained class whose physical/process parameters live entirely in its `__init__` (wire diameter, material thermophysics, power-source params, Goldak ellipsoid dimensions). Parameter studies are done by constructing with overrides — there is no config file. Modules are coupled only by passing scalar outputs between them (e.g. module 1's steady-state power drives module 4's heat source); they share no global state.

Two distinct simulation families, two entry points:

**Modules 1–5 (`main.py` → `welding-sim`):** classical lumped-parameter / PDE solvers.
- `gmaw.py` — `GMAWDynamics`: arc self-regulation ODE (stick-out `s`, current `I`), integrated with a CTWD step disturbance.
- `thermal.py` — `RosenthalThermal` (analytic moving point source) **and** `GoldakFDM` (3D transient double-ellipsoid FDM). The two are cross-validation of each other; `GoldakFDM` uses a half-symmetry domain with edge-pad Neumann boundaries and per-step heat-source renormalization for exact energy conservation. `GoldakFDM` is typically driven by `GMAWDynamics`' steady power.
- `droplet.py` — `DropletDynamics`: static-force-balance droplet transfer; `current_sweep` produces the globular→spray transition.
- `short_circuit.py` — `ShortCircuitGMAW`: hybrid arc⇄short-circuit state machine (standard short-circuit vs CMT mode).

**Modules 6–8 (`main_vi.py` → `welding-sim-vi`):** variational / geometric integrators built on a shared core.
- `variational.py` — the integrator toolkit: `ForcedVerlet` (symplectic Verlet + discrete Lagrange–d'Alembert forcing), `MidpointDEL` (midpoint discrete Euler–Lagrange with config-dependent mass matrix, Newton-solved), and nonsmooth collision-map utilities. Modules 6–8 are applications of these; changes here affect all three.
- `droplet_vi.py` — `DropletOscillatorVI`: Rayleigh l=2 droplet mode, pulsed-MIG resonance.
- `robot_vi.py` — `TwoLinkArm`: two-link welding-robot arm, config-dependent `M(q)`.
- `shortcircuit_vi.py` — `ContactCycleVI`: nonsmooth contact model of the CMT mechanical cycle (event-bisection + variational collision map + attach/retract/rupture state machine).

The central technical claim the VI modules demonstrate: symplectic/variational integrators keep energy error bounded over long trajectories where RK4 / implicit-Euler drift or add artificial damping. Preserve that property when touching `variational.py` (don't substitute a non-symplectic step).

`docs/legacy/` holds early single-file prototypes (`v0_*`, `v1_*`) kept for reference only — not imported by the package.

## Output convention

Both CLIs hardcode `matplotlib.use("Agg")` and write to `./results/` (created if missing) relative to the **current working directory** — run them from the repo root. They also print a one-line steady-state summary per module to stdout, which is the primary numerical check.
