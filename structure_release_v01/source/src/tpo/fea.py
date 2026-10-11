from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from tpo.mesh import Mesh


EquilibriumTerm = tuple[int, int, float]


def _free_dof_map(mesh: Mesh) -> dict[int, int]:
    return {int(global_dof): index for index, global_dof in enumerate(mesh.free_dofs)}


def _validate_element_ke(mesh: Mesh, element_ke: np.ndarray) -> np.ndarray:
    try:
        element_ke_array = np.asarray(element_ke, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("element_ke must be a finite NumPy-compatible array") from exc

    dofs_per_element = mesh.element_dofs.shape[1]
    expected_shape = (dofs_per_element, dofs_per_element)
    if element_ke_array.shape != expected_shape:
        raise ValueError(f"element_ke must have shape {expected_shape}")
    if not np.all(np.isfinite(element_ke_array)):
        raise ValueError("element_ke must contain only finite values")
    return element_ke_array


def _validate_stiffness_factors(mesh: Mesh, stiffness_factors: np.ndarray) -> np.ndarray:
    try:
        stiffness_factors_array = np.asarray(stiffness_factors, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("stiffness_factors must be a finite NumPy-compatible array") from exc

    if stiffness_factors_array.ndim != 1:
        raise ValueError("stiffness_factors must be one-dimensional")
    if stiffness_factors_array.size != mesh.element_count:
        raise ValueError(f"stiffness_factors length must be {mesh.element_count}")
    if not np.all(np.isfinite(stiffness_factors_array)):
        raise ValueError("stiffness_factors must contain only finite values")
    if np.any(stiffness_factors_array < 0.0):
        raise ValueError("stiffness_factors must be nonnegative")
    return stiffness_factors_array


def build_equilibrium_terms(mesh: Mesh, element_ke: np.ndarray) -> list[list[EquilibriumTerm]]:
    """Build free-DOF equilibrium terms for optimization constraints.

    Each term is ``(element_index, free_col_index, element_ke_value)``.
    Later Pyomo code multiplies the value by the element stiffness factor and
    the free displacement variable for ``free_col_index``.
    """
    element_ke_array = _validate_element_ke(mesh, element_ke)
    free_map = _free_dof_map(mesh)
    rows: dict[int, list[EquilibriumTerm]] = defaultdict(list)
    for element_index, dofs in enumerate(mesh.element_dofs):
        for local_row, global_row in enumerate(dofs):
            row_index = free_map.get(int(global_row))
            if row_index is None:
                continue
            for local_col, global_col in enumerate(dofs):
                col_index = free_map.get(int(global_col))
                if col_index is None:
                    continue
                value = float(element_ke_array[local_row, local_col])
                if value != 0.0:
                    rows[row_index].append((element_index, col_index, value))
    return [rows[i] for i in range(mesh.free_dofs.size)]


def assemble_free_stiffness(
    mesh: Mesh,
    element_ke: np.ndarray,
    stiffness_factors: np.ndarray,
) -> csr_matrix:
    element_ke_array = _validate_element_ke(mesh, element_ke)
    stiffness_factors_array = _validate_stiffness_factors(mesh, stiffness_factors)
    free_map = _free_dof_map(mesh)
    row_indices = []
    col_indices = []
    values = []
    for element_index, dofs in enumerate(mesh.element_dofs):
        factor = float(stiffness_factors_array[element_index])
        for local_row, global_row in enumerate(dofs):
            row_index = free_map.get(int(global_row))
            if row_index is None:
                continue
            for local_col, global_col in enumerate(dofs):
                col_index = free_map.get(int(global_col))
                if col_index is None:
                    continue
                row_indices.append(row_index)
                col_indices.append(col_index)
                values.append(factor * float(element_ke_array[local_row, local_col]))
    n = mesh.free_dofs.size
    return coo_matrix((values, (row_indices, col_indices)), shape=(n, n)).tocsr()


def solve_linear_response(
    mesh: Mesh,
    element_ke: np.ndarray,
    stiffness_factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = assemble_free_stiffness(mesh, element_ke, stiffness_factors)
    rhs = mesh.force[mesh.free_dofs]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            free_u = np.asarray(spsolve(matrix, rhs), dtype=float)
    except MatrixRankWarning as exc:
        raise RuntimeError("linear response solve failed: stiffness matrix is singular") from exc
    if not np.all(np.isfinite(free_u)):
        raise RuntimeError("linear response solve failed: non-finite displacement values")
    full_u = np.zeros(mesh.total_dofs, dtype=float)
    full_u[mesh.free_dofs] = free_u
    return free_u, full_u
