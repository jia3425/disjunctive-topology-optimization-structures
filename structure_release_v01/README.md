# Disjunctive topology optimization: structural code and data

Local review archive, version 01, prepared on 11 October 2026.

This package contains the existing structural optimization source code and archived numerical records for the cantilever, MBB and L-bracket comparisons, together with the separate cantilever continuation trajectory used in Figures 1c and 2e. It does not include the fluid, heat-transfer or membrane applications.

## Contents

- `source/src/tpo/`: 13 structural model, finite-element and supporting modules.
- `source/scripts/`: 7 original run, initialization and export scripts.
- `data/runs/`: 239 original numerical, configuration, history and solver-log files in seven run directories.
- `data/DATA_INDEX.json`: figure-to-result mapping, original paths and per-file SHA-256 checksums.
- `docs/DATA_GUIDE.md`: selected fields, stage status and interpretation notes.
- `docs/SOURCE_AND_ENVIRONMENT.md`: source provenance, recorded dependencies and path-adaptation requirements.
- `MANIFEST.sha256`: checksums for every package file except the manifest itself.

Start with the data guide to identify the selected result. Original data files are authoritative; edited presentation images are not included. No optimization or numerical-field modification was performed when preparing this archive.

## Inspecting saved data

NPZ files can be opened with NumPy without importing or running the optimization scripts:

```python
from pathlib import Path
import numpy as np

root = Path("structure_release_v01")
file = root / "data/runs/gdp_mbb_uniform_rerun_20260910_090145/solution.npz"
with np.load(file, allow_pickle=False) as result:
    print(result.files)
    solid_fraction = result["ys"]
    print(solid_fraction.shape, solid_fraction.min(), solid_fraction.max())
```

GDP phase fields use `ys`; the selected projected SIMP fields use `physical_density`. Array shape and mesh metadata should be read from the saved files rather than inferred from a rendered picture.

## Reproduction status

This is an archival snapshot, not a validated one-command reproduction package. Source and data copies have been checked against their originals. The code still contains original absolute paths and requires explicit adaptation on another machine. Ipopt must be installed separately. Three external SIMP initialization seeds and their earlier chains are documented but are not included in this snapshot. Some older runs retained scalar stage histories without all intermediate field snapshots. See both guides before attempting a rerun.

Recorded unsuccessful intermediate stages are retained for provenance; their inclusion does not assert convergence. The eight-stage cantilever comparison and the fourteen-stage illustrative trajectory are different runs. No end-to-end rerun, new physical validation or global-optimality claim is made by this archive.

## Access and rights

Intended distribution: a private GitHub repository and a separate Figshare private reviewer link, as approved by the author. This local package does not establish that either upload has completed, nor does it contain a DOI or an active reviewer link. Do not cite an archive URL until access has been verified.

No new license has been assigned. Any public release, license selection and third-party redistribution clearance require a separate author decision. No account credentials, repository history, solver binaries or manuscript Word files are included.
