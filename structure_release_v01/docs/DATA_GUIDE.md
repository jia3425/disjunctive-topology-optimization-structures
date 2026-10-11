# Structural data guide

This archive preserves six selected structural comparison runs and the separate 14-stage cantilever trajectory used by Figure 1c and Figure 2e. The seven run directories contain **239 copied files, 23,222,845 bytes (22.15 MiB)**. Files are exact copies: no optimization, new solver run, thresholding, hole filling, smoothing, or numerical-field editing was performed.

`data/DATA_INDEX.json` is the complete file inventory. For every copied file it records the original absolute path, archive-relative path, byte count, and SHA-256. Original run directory names are retained under `data/runs/`.

## Figure-to-result mapping

| Panel / method | Dataset ID | Selected field file relative to its run directory | Phase field | Reported compliance |
| --- | --- | --- | --- | ---: |
| Figure 2a and Figure 2d / SIMP | simp_cantilever | `converged_beta14/solution.npz` | physical_density | 208.33843408241825 |
| Figure 2a and Figure 2d / GDP | gdp_cantilever_8stage | `solution.npz` | ys | 207.33844799491334 |
| Figure 2b / SIMP | simp_mbb | `stage_07_beta_12/solution.npz` | physical_density | 227.21225460644035 |
| Figure 2b / GDP | gdp_mbb_uniform | `solution.npz` | ys | 225.13516627884113 |
| Figure 2c / SIMP | simp_lbracket | `stage_01_beta_12/solution.npz` | physical_density | 100.78958831838094 |
| Figure 2c / GDP | gdp_lbracket_original | `solution.npz` | ys | 100.08566088182933 |
| Figure 1c and Figure 2e / GDP trajectory | gdp_cantilever_14stage | See stage mapping below | ys | final: 209.1540444077738 |

Figure 1c selects `state.npz` from `stage_00_initial`, `stage_01`, `stage_04`, `stage_05`, `stage_07`, and `stage_14`. Figure 2e uses `stage_history.csv` and `stage_01/solution.npz` through `stage_14/solution.npz`, including explicitly unsuccessful intermediate records. Its compliance normalization is C_k/C_1, excluding the unsolved initial state. The complete available stage records are retained.

The Figure 2d raw sources are exactly those selected for Figure 2a. Its stress-recovery code and derived stress arrays are separate postprocessing artifacts, outside the seven raw-run directories copied here.

## Original run roots and copy counts

- **simp_cantilever** (14 files; 354,715 bytes): original `D:/Case/TPO/results/simp_projected_cantilever_60x20_20260909_181431`; archive `data/runs/simp_projected_cantilever_60x20_20260909_181431`.
- **gdp_cantilever_8stage** (7 files; 214,194 bytes): original `D:/Case/TPO/results/gdp_hull_relaxation_projected_weak_grayness_cantilever_60x20`; archive `data/runs/gdp_hull_relaxation_projected_weak_grayness_cantilever_60x20`.
- **simp_mbb** (19 files; 537,836 bytes): original `D:/Case/TPO/results/simp_projected_mbb_60x20_20260909_221209`; archive `data/runs/simp_projected_mbb_60x20_20260909_221209`.
- **gdp_mbb_uniform** (78 files; 8,189,415 bytes): original `D:/Case/TPO/results/gdp_mbb_uniform_rerun_20260910_090145`; archive `data/runs/gdp_mbb_uniform_rerun_20260910_090145`.
- **simp_lbracket** (8 files; 365,637 bytes): original `D:/Case/TPO/results/simp_projected_lbracket_60x60_p3_b12_oc_20260910_191100`; archive `data/runs/simp_projected_lbracket_60x60_p3_b12_oc_20260910_191100`.
- **gdp_lbracket_original** (7 files; 480,811 bytes): original `D:/Case/TPO/results/gdp_hull_relaxation_projected_weak_grayness_lbracket_60x60_uniform_p1_to_p2`; archive `data/runs/gdp_hull_relaxation_projected_weak_grayness_lbracket_60x60_uniform_p1_to_p2`.
- **gdp_cantilever_14stage** (106 files; 13,080,237 bytes): original `C:/Users/JQ/Documents/拓扑优化/paper/figures/figure_1/panel_c/solid_run_20260909`; archive `data/runs/solid_run_20260909`.

## Status and interpretation warnings

1. **Two distinct cantilever GDP runs.** Figure 2a/2d uses the old eight-stage run, with final compliance 207.33844799491334. Figure 1c/2e uses a different 14-stage uniform-start trajectory, with final compliance 209.1540444077738. Their terminal designs are not interchangeable. The old run contains scalar history and the final field, but no first-seven-stage field snapshots; none were synthesized.
2. **SIMP stage selection.** The cantilever beta-14 result is accepted; its later beta-16 stage reports `maxIterations` and `accepted=false`. The MBB beta-12 result is accepted; beta 14 reports `maxIterations` and `accepted=false`. These rejected stages remain in the archive as historical records. The selected L-bracket beta-12 NLP stage is accepted; `oc_initialization_only.npz` is only a warm start.
3. **Fourteen-stage cantilever records.** Stages 1-10 and 14 report `optimal`; stages 11-13 report `maxIterations`. Stage 12 grayness is 0.023112188906113, above its 0.02 limit. The unsolved `stage_00_initial` is not an equilibrium or volume-feasible solution. Intermediate unsuccessful records must not be represented as certified solutions.
4. **Original L-bracket GDP records.** The final zero-based stage index 8 reports `optimal`. Zero-based indices 5, 6, and 7 report `maxIterations`; index 7 grayness is 0.05956486305416422, above its 0.015 limit. Only final fields and scalar stage history are available in this run directory; missing intermediate fields were not reconstructed. The selected original result was not replaced by a later refined-route result.
5. **Figure 2c small hole is preserved.** The original GDP array has `ys[117] = 0.029527456840006083` (zero-based index, 2700 active elements), confirmed by a read-only array lookup. The complete copied NPZ matches the original SHA-256. No artificial filling or replacement field was used.
6. **Metadata flags require care.** Some original summaries say `completed_all_stages=true` even when intermediate stages reached iteration limits. This means the sequence was traversed, not that every stage converged. Original SIMP `run_config.json` files retain `status=running`; final interpretation comes from the stage summaries and histories. These original bytes were not corrected or rewritten.
7. **Numerical interpretation.** `optimal` is the recorded local NLP termination condition, not a proof of global optimality. The continuous physical phase fields are the selected data; rounded metrics in original summaries are separate diagnostics. Grayness values in the raw histories are fractions, so percentage displays multiply by 100. The SIMP and GDP exponents, initialization, and continuation schedules differ; these comparisons are not fully matched ablations.

## External seed dependencies not copied here

This copy task is limited to the seven named run directories. The original SIMP configurations reference the following earlier seeds outside that scope. Their listed hashes are copied as provenance from `run_config.json`; the external files themselves were not copied or freshly hash-verified by this data task. Their earlier seed chains may also be needed for full reruns.

- **simp_cantilever**: `D:/Case/TPO/results/simp_projected_cantilever_60x20_20260909_180758/stage_04_beta_8/solution.npz`; recorded SHA-256 `27c778a9ff01faea0e43c3880673b7b1b27a2405c8c93de8da5b5756237c9b1c`.
- **simp_mbb**: `D:/Case/TPO/results/simp_mbb_60x20/solution.npz`; recorded SHA-256 `2cdd06a2bdfeeec3ef60c8b35e08f9858b737464fe30af217975beb3dfd06711`.
- **simp_lbracket**: `D:/Case/TPO/results/simp_projected_lbracket_60x60_p3_b8_oc_20260910_191000/stage_01_beta_8/solution.npz`; recorded SHA-256 `f730ccdd23b76b19d9e68283d8d8fadfa8bce35d69756bd85151dfcaf907b473`.

## File types and integrity

The whitelist includes NPZ numerical arrays, JSON configuration/metadata/state records, CSV stage/metric tables, and solver or run stdout/stderr logs. Images, temporary files, caches, Python sources, and AI material are excluded. Available `start/state.*` and `stage_00_initial/state.*` records are retained, without implying that initial values satisfy equilibrium. Not every run recorded every file type.

All 239 source-to-copy SHA-256 comparisons passed during copying. The full per-file list is in `data/DATA_INDEX.json`. The index and this guide are new documentation, not original solver outputs. The hashes document byte preservation; they do not establish numerical correctness, end-to-end reproducibility, or successful convergence of every stored state. No side-effecting runner was imported, and no optimization or figure generation was executed.

