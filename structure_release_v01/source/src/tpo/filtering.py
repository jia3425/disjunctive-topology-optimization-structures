from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpo.mesh import Mesh


@dataclass(frozen=True)
class DensityFilter:
    rows: list[list[tuple[int, float]]]
    radius: float


def build_density_filter(mesh: Mesh, radius_elements: float) -> DensityFilter:
    if not np.isfinite(radius_elements) or radius_elements <= 0.0:
        raise ValueError("radius_elements must be a positive finite value")
    element_width = mesh.length / mesh.nelx
    radius = radius_elements * element_width
    rows: list[list[tuple[int, float]]] = []
    for center in mesh.centers:
        distances = np.linalg.norm(mesh.centers - center, axis=1)
        raw_weights = np.maximum(0.0, radius - distances)
        indices = np.flatnonzero(raw_weights > 0.0)
        total = float(raw_weights[indices].sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("density filter row has no positive finite weights")
        rows.append([(int(index), float(raw_weights[index] / total)) for index in indices])
    return DensityFilter(rows=rows, radius=radius)


def apply_density_filter(filter_data: DensityFilter, raw_density: np.ndarray) -> np.ndarray:
    raw_density = np.asarray(raw_density, dtype=float)
    if raw_density.ndim != 1:
        raise ValueError("raw_density must be one-dimensional")
    if raw_density.size != len(filter_data.rows):
        raise ValueError("raw_density length must match filter rows")
    if not np.all(np.isfinite(raw_density)):
        raise ValueError("raw_density must contain only finite values")
    physical = np.zeros(len(filter_data.rows), dtype=float)
    for row_index, row in enumerate(filter_data.rows):
        physical[row_index] = sum(weight * raw_density[col] for col, weight in row)
    return physical
