# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A multiphysics dynamics simulation package for GMAW/MIG arc welding, written in Python (NumPy/SciPy/Matplotlib). It models the full chain from arc circuit → droplet transfer → workpiece heat conduction, plus a variational-integrator extension for oscillation/contact problems. There is no web/service layer — it is a pure simulation library with two CLI entry points that produce PNG plots.

Note: the README and most code comments/docstrings are in Chinese. Match that convention when editing existing modules.

## Commands

```bash
uv sync             # create .venv and install deps (numpy, scipy, matplotlib, hydra-core)
uv sync --extra viz # additionally install mayavi (heavy: VTK/Qt) for module 9 rendering
uv sync --extra notebook  # mayavi + jupyter for the interactive demo notebook
uv run welding-sim      # run modules 1–5, writes plots to ./results/
uv run welding-sim-vi   # run variational modules 6–8, writes plots to ./results/
uv run welding-sim-3d   # module 9: solve GoldakFDM, export OpenFOAM case, Mayavi render
uv run jupyter lab notebooks/mayavi_interactive_demo.ipynb  # interactive 3D render
```

All three CLIs are Hydra apps (`conf/` inside the package). Override from the command line:

```bash
uv run welding-sim --cfg job              # print the composed config, run nothing
uv run welding-sim process=db_median      # swap a config group
uv run welding-sim material=aluminum solver=fine gmaw.Voc=30.0   # group + leaf override
uv run welding-sim --multirun process=db_p10,db_median,db_p90 output=per_run  # sweep
```

There is **no test suite, linter, or formatter configured.** Validation is done by running the two CLIs and inspecting the printed steady-state numbers (compared against the reference table in README.md) and the regenerated `results/*.png`. When changing physics, run the relevant CLI **with default config groups** and confirm the printed values still match the README's "典型结果" table — the defaults (`process=code_default material=carbon_steel solver=default`) are pinned to reproduce that table exactly.

To exercise a single module without running everything, import the class directly. The classes take plain kwargs and **do not depend on Hydra** — the config tree is a layer on top, not a requirement:

```python
uv run python -c "from welding_dynamics import GMAWDynamics; print(GMAWDynamics().simulate()['I'][-1])"
```

## Architecture

The package (`src/welding_dynamics/`) is a set of independent physics modules, each a self-contained class whose physical/process parameters live entirely in its `__init__` (wire diameter, material thermophysics, power-source params, Goldak ellipsoid dimensions). Every class takes plain keyword arguments and can be constructed directly with overrides. Modules are coupled only by passing scalar outputs between them (e.g. module 1's steady-state power drives module 4's heat source); they share no global state.

**Configuration (`conf/`, Hydra).** The three CLIs compose their parameters from a Hydra config tree at `src/welding_dynamics/conf/` (needs `conf/__init__.py` — Hydra resolves a package-relative `config_path` as an importable module). It is a *layer over* the classes, not a replacement: nothing in the library imports Hydra, and `GoldakFDM(Q=9000)` still works standalone.

- Config groups mirror the three kinds of parameter: `process/` (operating point — current, arc power, travel speed, CTWD, wire diameter; the quantities a welding-procedure database can supply), `material/` (handbook thermophysics), `solver/` (grid `dx`, domain, `t_end` — numerics only), `weave/` (torch weaving for `GoldakFDM`: `none` default, `triangle`/`sine` analytic, `pattern1` a 摆动库 waypoint waveform; only `sim`/`sim_3d` have this group), `output/` (plot dir + dpi).
- `model/*.yaml` are `_target_` nodes, one per class, built with `hydra.utils.instantiate`. Physical constants appear **once**: model nodes interpolate (`${material.k}`) rather than copying. Derived quantities use the custom resolvers in `config.py` — `${wd.half:...}` (diameter→radius) and `${wd.alpha:k,rho,cp}` (thermal diffusivity), so `alpha` can't drift from `k/(rho*cp)`.
- `process.arc_power_W: null` means "take the upstream power": in `welding-sim` that's module 1's steady `P_ss`; in `welding-sim-3d` it means "don't pass `Q`", i.e. fall back to `GoldakFDM`'s class default. The `db_*` presets set it explicitly so the thermal model is driven by measured `U·I`. Resolve it via `config.arc_power(cfg, fallback=...)`, never by reading the field directly.
- `hydra.job.chdir` is **false** in every root config, so `./results/` stays relative to the repo root as the README promises. Hydra's own run dirs (`results/runs/`, `results/multirun/`) hold the config snapshot + log and are gitignored. For sweeps use `output=per_run`, which points `output.dir` at `${hydra:runtime.output_dir}` so parallel combinations don't overwrite each other's PNGs.
**MongoDB stores (`project_data/`, optional).** Two collections in db `welding_dynamics`, each rebuilt idempotently by its own script (both `drop()` then re-insert, and use a `doc_type` discriminator). Neither is needed to run the simulations.

```bash
uv run python project_data/ingest_mongo.py           # -> welding_parameters (工艺数据库 xlsx)
uv run python project_data/ingest_config_mongo.py    # -> welding_config     (conf/ 配置树)
uv run python project_data/ingest_config_mongo.py --dry-run   # 只打印, 不写库
```

`welding_config` holds `config_root` (3 root yamls), `config_group` (21 group options), `config_composed`, and `config_meta` (git commit, hydra version, per-file sha256). The `config_composed` docs are the payoff: each is a **fully composed and interpolation-resolved** snapshot (`${material.k}`, `${wd.alpha:...}` already evaluated) for one `(root, process)` combination, queryable via `groups.process`. A stored `resolved` dict round-trips: `instantiate(OmegaConf.create(doc["resolved"]).goldak, Q=...)` rebuilds the object. Note `output=per_run` is deliberately *not* composed — its `${hydra:runtime.output_dir}` only resolves inside a live Hydra run.

- The `db_*` process presets come from the production parameter database (see `notebooks/welding_parameter_database_exploration.ipynb`, §9). Caveat baked into `main.py`: the database records **no wire-feed speed**, and module 1's working point is set by `(WFS, CTWD)` — so its steady current need not equal the preset's nominal `process.current_A`. `main.py` prints a `[1 提示]` warning when they differ by >10%. `current_A`/`voltage_V` are documentation-only fields; only `arc_power_W`, `travel_speed_m_s`, `ctwd_m`, `wire_diameter_m` actually drive anything.

Two distinct simulation families, two entry points:

**Modules 1–5 (`main.py` → `welding-sim`):** classical lumped-parameter / PDE solvers.
- `gmaw.py` — `GMAWDynamics`: arc self-regulation ODE (stick-out `s`, current `I`), integrated with a CTWD step disturbance.
- `thermal.py` — `RosenthalThermal` (analytic moving point source) **and** `GoldakFDM` (3D transient double-ellipsoid FDM). The two are cross-validation of each other; `GoldakFDM` uses a half-symmetry domain with edge-pad Neumann boundaries and per-step heat-source renormalization for exact energy conservation. `GoldakFDM` is typically driven by `GMAWDynamics`' steady power. Passing `weave=` (a `weave.py` object) gives the heat source a lateral offset `y_s(t)`; since that breaks the y=0 mirror symmetry, the constructor then switches to a **full-width grid** (`Ny = 2*Ly/dx - 1`, y=0 on a cell center, `j_center` marks the centerline, ~2× cost). Beware: the half model's edge-pad mirror sits at the face `y=-dx/2`, the full grid mirrors through the y=0 cell center — so pointwise T near the axis differs O(dx) between the two even at zero weave amplitude (pool L/W/D and total energy match; don't chase bit-identity).
- `weave.py` — torch-weaving motion models consumed by `GoldakFDM(weave=...)`: `HarmonicWeave` (sine/triangle, `amplitude_m` is **peak-to-peak**) and `WaypointWeave` (摆动库 `weave_pattern` waypoint waveforms, normalized so max |y_pct| maps to A/2; vertical `z_pct` deliberately not modeled). Weave frequency (DB range 0.3–4 Hz) is ~2 orders below the droplet Rayleigh mode (~527 Hz), so weaving moves the heat source but does not couple into module 6.
- `droplet.py` — `DropletDynamics`: static-force-balance droplet transfer; `current_sweep` produces the globular→spray transition.
- `short_circuit.py` — `ShortCircuitGMAW`: hybrid arc⇄short-circuit state machine (standard short-circuit vs CMT mode).

**Modules 6–8 (`main_vi.py` → `welding-sim-vi`):** variational / geometric integrators built on a shared core.
- `variational.py` — the integrator toolkit: `ForcedVerlet` (symplectic Verlet + discrete Lagrange–d'Alembert forcing), `MidpointDEL` (midpoint discrete Euler–Lagrange with config-dependent mass matrix, Newton-solved), and nonsmooth collision-map utilities. Modules 6–8 are applications of these; changes here affect all three.
- `droplet_vi.py` — `DropletOscillatorVI`: Rayleigh l=2 droplet mode, pulsed-MIG resonance.
- `robot_vi.py` — `TwoLinkArm`: two-link welding-robot arm, config-dependent `M(q)`.
- `shortcircuit_vi.py` — `ContactCycleVI`: nonsmooth contact model of the CMT mechanical cycle (event-bisection + variational collision map + attach/retract/rupture state machine).

The central technical claim the VI modules demonstrate: symplectic/variational integrators keep energy error bounded over long trajectories where RK4 / implicit-Euler drift or add artificial damping. Preserve that property when touching `variational.py` (don't substitute a non-symplectic step).

**Module 9 (`main_3d.py` → `welding-sim-3d`):** 3D post-processing of the `GoldakFDM` field, in `thermal3d.py`.
- `OpenFOAMExporter` (pure numpy, no mayavi) — writes the FDM structured grid as a hand-built OpenFOAM `polyMesh` (points/faces/owner/neighbour/boundary, upper-triangular face ordering; half-symmetric runs get a `symmetryPlane` patch at y=0, full-width weave runs fold the yMin side into `farField` instead — patches and `boundaryField` entries both follow `fdm.symmetric`) plus `T`/`Tpeak` as `volScalarField` time directories and a minimal runnable `laplacianFoam` `system/`+`constant/`. Cell↔array mapping is `field.ravel(order="F")` for the `[i,j,k]` array (cell index `i + Nx*(j + Ny*k)`). Drops a `case.foam` stub so the case opens directly in ParaView. **If you change the mesh writer, re-validate closedness** — every cell's summed face-area vectors must be ~0 (this is what catches a wrong point ordering or owner/neighbour flip); there is no OpenFOAM install to run `checkMesh`.
- `render()` — Mayavi (`mlab`) volumetric view: mirrors the half-model across y=0 (full-width weave fields are used as-is, no mirroring — `_full_field` branches on `fdm.symmetric`), draws melt/HAZ iso-surfaces + a symmetry-plane slice. **mayavi is an optional dependency** (`[project.optional-dependencies] viz`), lazy-imported inside `render()`; the exporter and the rest of the package work without it. The CLI degrades gracefully (prints and continues) when mayavi is absent or offscreen rendering fails headless. `render(..., notebook=True)` returns the figure for inline Jupyter display instead of calling the blocking `mlab.show()` — used by `notebooks/mayavi_interactive_demo.ipynb` (requires `mlab.init_notebook()` first; `notebook` optional-deps group adds jupyter + interactive backends).

`docs/legacy/` holds early single-file prototypes (`v0_*`, `v1_*`) kept for reference only — not imported by the package.

## Output convention

Both CLIs hardcode `matplotlib.use("Agg")` and write to `./results/` (created if missing) relative to the **current working directory** — run them from the repo root. They also print a one-line steady-state summary per module to stdout, which is the primary numerical check.
