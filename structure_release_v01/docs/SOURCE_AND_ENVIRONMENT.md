# Structural source archive and environment

This document describes the local source archive prepared on 2026-10-11. The copied Python files are byte-for-byte copies of the existing files at the original paths listed below. No algorithm, parameter default, solver implementation, or saved numerical result was modified during archiving. No optimization was run.

## Scope and roles

The source archive contains 20 Python files (226,982 bytes): 13 package modules in `source/src/tpo/` and 7 scripts in `source/scripts/`. The package subset follows the recursive `tpo` imports of the selected structural scripts, including imports inside functions, plus the package initializer. Fluid modules, caches, solver binaries, system fonts, account configuration, and repository history are excluded.

- `gdp_hull_relaxation.py` and `gdp_hull_relaxation_io.py` implement the structural relaxed GDP model, continuation, and saved solution/summary output. `gdp_hull_relaxation_diagnostics.py` provides the associated diagnostics.
- `gdp_hull_binary.py` is required by the relaxation module for shared formulation utilities. Its inclusion is an import dependency and does not imply that the reported structural results were obtained by a global binary solve.
- `mesh.py`, `element.py`, `fea.py`, and `filtering.py` provide benchmark geometry, Q4 stiffness, finite-element response, and density filtering.
- `simp_model.py` implements the SIMP comparison model; `io.py` and `plotting.py` provide structural output helpers; `runtime.py` configures the original local solver.
- `run_gdp_hull_relaxation_projected_weak_grayness.py` is the parameterized legacy GDP continuation entry point.
- `run_solid.py` is the recorder used for the Figure 1c / Figure 2e initial state and 14-stage structural trajectory.
- `run_gdp_uniform_paper.py` imports that recorder dynamically and supports the fresh uniform-start MBB route and a later refined L-bracket route. The presence of a supported route in this script does not identify it as the selected Figure 2c result; the archive's figure-to-result mapping governs selection.
- `run_simp_projected_paper.py` performs projected SIMP continuation. `projected_simp_oc_seed.py` supplies an optional OC warm start; final acceptance belongs to the full-space NLP.
- `run_simp_benchmark.py` is the original structural SIMP benchmark entry point relevant to warm-start provenance.
- `export_simp_projected_converged.py` records the accepted beta-14 cantilever export procedure. It executes file operations at module top level and must not be imported as a passive utility.

Use saved per-run configurations, stage histories, summaries, and figure provenance to identify the parameters and result selected for a paper panel. Current script defaults alone are not a complete historical command record.

## Recorded original runtime

The original environment record is `D:/Case/TPO/PROJECT_RUNTIME_ENVIRONMENT.md`. It reports the following; versions were not reinstalled or freshly validated in this archive task.

| Component | Recorded version or setting |
| --- | --- |
| Operating system / shell | Windows 10, build 26200 / PowerShell |
| Python | 3.11.7, Anaconda base environment |
| Pyomo | 6.10.1 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| pandas | 2.3.3 (recorded in the broader environment) |
| idaes-pse | 2.11.0 (original solver distribution route) |
| Ipopt | 3.13.2 |
| Linear solver used by the structural runs | MUMPS |
| Other packages in the broader recorded environment | PySCIPOpt 6.2.1; Bonmin 1.8.8 and Couenne 0.5.8 were also available |

The selected Python sources use NumPy, SciPy, Pyomo, and Matplotlib in addition to the standard library. The original record does not pin Matplotlib. Availability of other optimization packages in the broader environment does not establish that they were used by the archived structural runs.

Some original figure-rendering scripts outside this source subset use Node.js and `sharp`; the editable Figure 1c scene uses Blender. These are separate presentation dependencies. Neither fonts nor solver executables are redistributed.

## Portability and limits of validation

This is a source-and-results archive, not a newly validated portable installation. Hard-coded paths have intentionally been preserved so that copied source hashes remain identical to the existing originals.

Before execution on another machine, a separate working copy would need explicit path adaptation for:

1. The project, source, and results roots under `D:/Case/TPO`.
2. The Ipopt executable path in `source/src/tpo/runtime.py`; installing Pyomo alone does not provide that executable.
3. The absolute `HELPER_PATH` in `run_gdp_uniform_paper.py`, which points to the original `run_solid.py`. The archived copy is now under `source/scripts/`.
4. Original data, figure, comparison, and protection-hash paths in the scripts, including the Figure 1c / Figure 2 directories.
5. Local font paths used by some plotting routines. Separate original rendering scripts also contain a local `sharp` package path.

Warm-started SIMP runs depend on earlier saved seed fields. The requested benchmark, complete continuation schedule, initialization chain, solver options, and stopping conditions must be reconstructed from the saved run records, not inferred from the last `solution.npz` alone. Some historical GDP directories retain only final fields and stage scalar records; the absence of earlier field snapshots must not be concealed by creating synthetic fields.

Several scripts create result directories or write output relative to their own location or the original absolute paths. Review those destinations before any future execution, and execute only against an explicitly prepared working copy with a separate output directory.

Archive validation in this task consisted of dependency-oriented source selection and SHA-256 equality checks between every copied file and its original. No solver, model import, optimization, end-to-end run, numerical reproduction, or figure regeneration was performed. Successful byte comparison establishes copy integrity; it does not establish end-to-end reproducibility, convergence of every saved stage, or global optimality.

No new license has been assigned. Code ownership and any third-party attribution or redistribution requirements remain to be resolved before public distribution.

## Original file paths and SHA-256

All archive paths in this table are relative to `structure_release_v01/`. Hashes are SHA-256 of the exact source bytes, also verified against the copied bytes.

| Archive path | Original path | SHA-256 |
| --- | --- | --- |
| `source/scripts/run_gdp_hull_relaxation_projected_weak_grayness.py` | `D:/Case/TPO/scripts/run_gdp_hull_relaxation_projected_weak_grayness.py` | `b6f0d1ce4b3da9584edae8bf99f02c89d6629cf804913ffa99e19d5b60d7c889` |
| `source/scripts/run_gdp_uniform_paper.py` | `D:/Case/TPO/scripts/run_gdp_uniform_paper.py` | `89c5022b8065bd21e780473b32883a290e7d6a341c197794f2d9e28e744b1211` |
| `source/scripts/run_simp_projected_paper.py` | `D:/Case/TPO/scripts/run_simp_projected_paper.py` | `97d04102a9e0a9a02a5e97430393cc039248e83c8c9d7af3428eca6b1541e121` |
| `source/scripts/projected_simp_oc_seed.py` | `D:/Case/TPO/scripts/projected_simp_oc_seed.py` | `4decd36f0a8f8ae330181fb746fd8823ab1b397a77b1fb2acd5d2c55e7cc4898` |
| `source/scripts/export_simp_projected_converged.py` | `D:/Case/TPO/scripts/export_simp_projected_converged.py` | `55b4c7901a373350163e5fc694311e5e3c74c8c931b620d13fd46c7165f43b28` |
| `source/scripts/run_simp_benchmark.py` | `D:/Case/TPO/scripts/run_simp_benchmark.py` | `799440d87d0815870f966b5925d8a5a313cc9c189a06c5a16c2b82471f7e4abb` |
| `source/scripts/run_solid.py` | `C:/Users/JQ/Documents/拓扑优化/paper/figures/figure_1/panel_c/solid_run_20260909/run_solid.py` | `39051e6ab187370f72257da7203d43a21ba80714ca8b318ff01f3fc7c7e9bde4` |
| `source/src/tpo/gdp_hull_relaxation.py` | `D:/Case/TPO/src/tpo/gdp_hull_relaxation.py` | `1e779a24666dab4680b16182917f9d7ee85b325b5a31a00f356eaf6b149fad7b` |
| `source/src/tpo/gdp_hull_relaxation_io.py` | `D:/Case/TPO/src/tpo/gdp_hull_relaxation_io.py` | `78f5d83bb3a6a25ebb2721b578ed74a836f3e32796f0582f498433c41dba6d18` |
| `source/src/tpo/gdp_hull_relaxation_diagnostics.py` | `D:/Case/TPO/src/tpo/gdp_hull_relaxation_diagnostics.py` | `f57daa8c4309d8ff7326ff472d07f0c172091e8160cf3e3a71e8c8e2a4893a8a` |
| `source/src/tpo/plotting.py` | `D:/Case/TPO/src/tpo/plotting.py` | `10930842c378bec317a7df83a53c59e6e537e113ab21d2757213ea0a2b15afeb` |
| `source/src/tpo/runtime.py` | `D:/Case/TPO/src/tpo/runtime.py` | `119c06cba89e7c3bdb744855b1384e40879cddf1495d9566d93ae1e7f7d96d2a` |
| `source/src/tpo/fea.py` | `D:/Case/TPO/src/tpo/fea.py` | `034f01e4efa2b6c069277d0b3fd5031afddfaa69b9100b57b44afcfd64380a34` |
| `source/src/tpo/simp_model.py` | `D:/Case/TPO/src/tpo/simp_model.py` | `c64fcfd70aae14f04c45f0f77140680e4e79169efb7fec287d6936817114befe` |
| `source/src/tpo/filtering.py` | `D:/Case/TPO/src/tpo/filtering.py` | `e4fd988dda0f183106cdd25de236c86be0732e415d2d9bdc478183ae085dc233` |
| `source/src/tpo/io.py` | `D:/Case/TPO/src/tpo/io.py` | `af18406a07288789842f3c259cf407903a337e85308c3e8d9558579528f7b3bf` |
| `source/src/tpo/element.py` | `D:/Case/TPO/src/tpo/element.py` | `5c0988ca29de9c1780b01a41a185cd225370038241add8ff9952ef20017b116f` |
| `source/src/tpo/gdp_hull_binary.py` | `D:/Case/TPO/src/tpo/gdp_hull_binary.py` | `05962415a32d5e30e928564e0630d4cb4a1090d67a72083f8429c660f8dbab36` |
| `source/src/tpo/mesh.py` | `D:/Case/TPO/src/tpo/mesh.py` | `0ef16b02d0de7e9099997a21c8c2239580838867448241852d55c2381018357f` |
| `source/src/tpo/__init__.py` | `D:/Case/TPO/src/tpo/__init__.py` | `cd14e144c5b111ca45ea70b123d647338bcc11c892da9edd36a0347426164367` |

