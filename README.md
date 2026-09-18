# variational-poroelasticity-model

Code and data for ***A variational model of nonlinear poroelasticity*** (Adler, Hu,
Kirshtein).

This repository contains only what is needed to produce the results reported in the
paper: the two-field finite-element solver, the three experiment drivers behind the
figures, the nine footing configurations, the result data behind every figure, and a
notebook that redraws every computed figure inline.

## Contents

```
porousflow.py            shared solver library (FEniCSx): spaces, free energy omega and
                         its derivatives, elastic stress, energy functionals
run_accuracy.py          Figure 1   — manufactured-solution convergence
run_energy_blob.py       Figure 3   — energy decay and discrete energy-law defect
run_footing.py           Figures 5-8, Table 1 — footing consolidation, one config per run
configs/                 the nine footing configurations (3 side treatments x 3 gamma)
data/                    result files small enough to ship (see below)
paper_figures.ipynb      redraws Figures 1, 3, 5, 6-8 and Table 1 inline
```

### What maps to what

| Paper item | Produced by | Needs |
|---|---|---|
| Figure 1 — accuracy | `run_accuracy.py` | bundled |
| Figure 3 — energy behaviour | `run_energy_blob.py` | bundled |
| Figure 5 — footing stored energy | `run_footing.py` (9 configs) | bundled |
| Table 1 — footing summary | `run_footing.py` (9 configs) | bundled |
| Figures 6–8 — deformed domain | `run_footing.py` (9 configs) | bundled (5 of the ~200 saved times per config) |

Figures 2 and 4 are TikZ schematics drawn in the manuscript itself. They involve no
computation and have no code here.

## Requirements

FEniCSx (DOLFINx 0.10) with PETSc, plus NumPy. The notebook additionally needs
Matplotlib, and the `vtk` Python package to read the deformed-domain snapshots for
Figures 6–8. With conda:

```bash
conda create -n fenicsx -c conda-forge fenics-dolfinx petsc4py mpi4py numpy matplotlib vtk
conda activate fenicsx
```

Running the solver (`run_accuracy.py`, `run_energy_blob.py`, `run_footing.py`) needs the
FEniCSx/PETSc part only; the notebook needs the rest.

## Reproducing the results

All three drivers write into `data/` (the accuracy and blob drivers) or `output/`
(the footing driver) next to this file. Runtimes are single-core, on a workstation.

### Figure 1 — accuracy (~1 hour)

```bash
python run_accuracy.py
```

Runs both sweeps at the benchmark parameters `E=1, nu=0.2, kappa=1, M=alpha=1`:

* **spatial** `K = 8, 16, 32` at `dt = 1e-6`, `T = 5e-6`. The time step is far below
  `h^2/(kappa*pi^2)`, so the temporal error stays below the spatial error.
* **temporal** `dt = 0.01, 0.005, 0.0025` at `K = 256`, `T = 0.1`. The mesh is chosen so
  the *spatial* error stays well below the temporal error across this range — at coarser
  meshes the fixed spatial error becomes a floor and the measured rate degrades. This is
  the expensive part (~45 s/step at `K=256`).

Errors are reported in the energy quantities of Section 4.1 — the elastic energy norm
for displacement and the Bregman divergence of the fluid free energy for density —
alongside the `L2` and `H1`-seminorm values for reference. Expected rates: `2.04, 2.01`
and `2.02, 2.00` (spatial), `1.99, 1.89` and `2.05, 2.22` (temporal).

### Figure 3 — energy behaviour (~10 minutes)

```bash
python run_energy_blob.py            # both parts
python run_energy_blob.py timeseries # just the decay curve
python run_energy_blob.py defect     # just the defect sweep
```

Sealed blob, all walls clamped, no forcing. The `timeseries` part (`K=32`, `dt=1e-3`,
`T=0.5`) gives the energy decay to ~96.6% of its initial value; the `defect` part
(`K=64`, `T=0.5`, `dt = 0.05, 0.02, 0.01, 0.005`) gives `DEFECT(T)` converging at
`O(dt^2)` (rates ~1.86, 1.94, 2.01).

### Figures 5–8 and Table 1 — footing (long; ~43 GB of output)

One configuration per invocation:

```bash
for cfg in configs/footing_*.txt; do python run_footing.py "$cfg"; done
```

Each run is `K=128`, `dt=1e-3`, `T=2`, ramp-then-hold load to `p_max=100` over
`T_ramp=1`, at the footing parameters `E=1000, nu=0.2, kappa=1e-2`. Per configuration it
writes `history.csv` (load, stored energy, fluid mass, Newton iteration counts — this is
all Figure 5 and Table 1 need, ~400 kB) and ~200 VTK snapshots (~5 GB, of which
`data/footing_vtk/` bundles only the 5 the deformed-domain figures actually show).

Set `PF_OUTPUT_ROOT` to redirect the output somewhere with room.

## Figures

```bash
jupyter lab paper_figures.ipynb
```

Every figure and Table 1 run against the bundled data with no setup. To browse a footing
run's full time resolution instead of the 5 bundled snapshots, point `PF_VTK_ROOT` at its
output directory:

```bash
PF_VTK_ROOT=/path/to/output/footing jupyter lab paper_figures.ipynb
```

## Bundled data

| Path | Contents |
|---|---|
| `data/accuracy/spatial.csv`, `temporal_K256.csv` | Figure 1 |
| `data/energy_blob/energy_vs_time.csv`, `defect_vs_dt.csv` | Figure 3 |
| `data/footing/{clamped,sliding,free}_g{1,2,5}.csv` | Figure 5, Table 1 |
| `data/footing_vtk/{clamped,sliding,free}_g{1,2,5}/` | Figures 6–8 |

The first three are the `history.csv` / summary files from the runs reported in the
paper, ~3.5 MB total. `data/footing_vtk/` carries the displacement (`u`), Darcy flux
(`q`), and density (`rho`) fields at the 5 times $t=0,0.5,1,1.5,2$ each deformed-domain
panel shows, for all 9 configurations (~650 MB) — a slice of the ~5 GB of VTK output
each configuration writes in full, not the full time series.

Note that the three side treatments are named `clamped`, `sliding`, `free` in the code
and **fixed**, **roller**, **free** in the paper.

## Scope

`porousflow.py` is the shared solver library. Besides the two-field spaces and
functionals used here, it also carries code paths for a three-field (`RT`/`DG`)
formulation and for mixed single-block solves, which this paper does not use.

## Citing

If you use this code, please cite the paper, and the software itself if you are
reporting reproductions of these results. Machine-readable metadata is in
`CITATION.cff`.

> J. H. Adler, X. Hu, A. Kirshtein, *A variational model of nonlinear poroelasticity*.

The paper is not yet published; the full reference and DOI will be added here, and to
`CITATION.cff`, once it appears.

## License

MIT — see [`LICENSE`](LICENSE).

The dependencies keep their own licences: DOLFINx, FFCx and UFL are LGPL-3.0-or-later,
Basix is MIT, PETSc is BSD-2-Clause. Importing them imposes no additional conditions on
this code, but redistributing them does.
