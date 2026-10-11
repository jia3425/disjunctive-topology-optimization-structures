# Disjunctive topology optimization — structural code and data

Private archival repository for the structural examples accompanying the manuscript. Archive version 01 was prepared on 11 October 2026. The original source and numerical files are preserved byte-for-byte; no optimization or numerical-field editing was performed during packaging.

## Browse or download

- [Complete review package (ZIP, 19.1 MB)](archive/structure_release_v01.zip)
- [ZIP SHA-256 checksum](archive/structure_release_v01.zip.sha256)
- [Package overview and data-reading example](structure_release_v01/README.md)
- [Source code](structure_release_v01/source/)
- [Original numerical data and solver records](structure_release_v01/data/runs/)
- [Figure-to-result mapping and data guide](structure_release_v01/docs/DATA_GUIDE.md)
- [Source provenance and recorded environment](structure_release_v01/docs/SOURCE_AND_ENVIRONMENT.md)
- [Complete data index](structure_release_v01/data/DATA_INDEX.json)
- [Per-file SHA-256 manifest](structure_release_v01/MANIFEST.sha256)

The package contains 20 Python source files and 239 original data, configuration, history and solver-log files. It covers the cantilever, MBB and L-bracket structural comparisons and the separate cantilever continuation trajectory for Figures 1c and 2e. Fluid, heat-transfer and membrane examples are not part of this package.

## Reproduction and access status

This is an archival snapshot, not a validated one-command reproduction package. Original absolute paths remain in the code and require adaptation before execution on another machine. Three external SIMP initialization seeds and their earlier chains are documented but not included. Some older runs lack intermediate field snapshots. Recorded unsuccessful intermediate stages are retained as historical records, not certified solutions. Read the linked guides before attempting a rerun.

This repository is private: its URL alone does not grant access to an unauthorised account. A separate Figshare private reviewer link is planned but has not been created. No DOI or anonymous reviewer-access URL is claimed here. No new license has been assigned, and no solver binaries are redistributed.

## Integrity

Expected SHA-256 of `archive/structure_release_v01.zip`:

```text
41de0ae2eaa6acac20c04a6019e8e732a71426c49e9089420003274f7007d841
```

The ZIP contains the same `structure_release_v01/` directory that can be browsed in this repository. Checksums cover the preserved package, not this repository landing page or Git metadata.
