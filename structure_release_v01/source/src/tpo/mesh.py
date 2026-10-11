from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    nelx: int
    nely: int
    length: float
    height: float
    nodes: np.ndarray
    elements: np.ndarray
    element_dofs: np.ndarray
    centers: np.ndarray
    fixed_dofs: np.ndarray
    free_dofs: np.ndarray
    force: np.ndarray
    element_area: float
    element_grid_indices: np.ndarray | None = None
    active_element_mask: np.ndarray | None = None

    @property
    def total_dofs(self) -> int:
        return self.nodes.shape[0] * 2

    @property
    def element_count(self) -> int:
        return self.elements.shape[0]


def element_values_to_grid(
    mesh: Mesh,
    values,
    fill_value: float = np.nan,
    name: str = "element values",
) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite NumPy-compatible array") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size != mesh.element_count:
        raise ValueError(f"{name} length must be {mesh.element_count}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")

    if mesh.element_grid_indices is None:
        if mesh.element_count != mesh.nelx * mesh.nely:
            raise ValueError("mesh.element_grid_indices is required for inactive domains")
        return vector.reshape(mesh.nely, mesh.nelx)

    grid_indices = np.asarray(mesh.element_grid_indices, dtype=int)
    if grid_indices.shape != (mesh.element_count, 2):
        raise ValueError(f"mesh.element_grid_indices shape must be {(mesh.element_count, 2)}")

    field = np.full((mesh.nely, mesh.nelx), fill_value, dtype=float)
    for value, (ey, ex) in zip(vector, grid_indices):
        if ey < 0 or ey >= mesh.nely or ex < 0 or ex >= mesh.nelx:
            raise ValueError("mesh.element_grid_indices contains an out-of-range cell")
        field[ey, ex] = value
    return field


def _node_id(ix: int, iy: int, nelx: int) -> int:
    return iy * (nelx + 1) + ix


def _validate_geometry(
    nelx: int,
    nely: int,
    length: float,
    height: float,
) -> None:
    if nelx <= 0 or nely <= 0:
        raise ValueError("nelx and nely must be positive")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be a positive finite value")
    if not np.isfinite(height) or height <= 0:
        raise ValueError("height must be a positive finite value")


def _build_nodes(nelx: int, nely: int, length: float, height: float) -> np.ndarray:
    dx = length / nelx
    dy = height / nely

    nodes = []
    for iy in range(nely + 1):
        for ix in range(nelx + 1):
            nodes.append((ix * dx, iy * dy))
    return np.asarray(nodes, dtype=float)


def _active_element_nodes(ex: int, ey: int, nelx: int) -> list[int]:
    return [
        _node_id(ex, ey, nelx),
        _node_id(ex + 1, ey, nelx),
        _node_id(ex + 1, ey + 1, nelx),
        _node_id(ex, ey + 1, nelx),
    ]


def _build_structured_mesh(
    nelx: int,
    nely: int,
    length: float,
    height: float,
    active_element_mask: np.ndarray,
    fixed_dofs,
    force: np.ndarray,
) -> Mesh:
    _validate_geometry(nelx, nely, length, height)
    dx = length / nelx
    dy = height / nely
    active_mask = np.asarray(active_element_mask, dtype=bool)
    if active_mask.shape != (nely, nelx):
        raise ValueError(f"active_element_mask shape must be {(nely, nelx)}")

    node_array = _build_nodes(nelx, nely, length, height)

    elements = []
    centers = []
    element_dofs = []
    element_grid_indices = []
    for ey in range(nely):
        for ex in range(nelx):
            if not active_mask[ey, ex]:
                continue
            elem_nodes = _active_element_nodes(ex, ey, nelx)
            elements.append(elem_nodes)
            centers.append(((ex + 0.5) * dx, (ey + 0.5) * dy))
            element_grid_indices.append((ey, ex))
            dofs = []
            for node in elem_nodes:
                dofs.extend([2 * node, 2 * node + 1])
            element_dofs.append(dofs)
    if not elements:
        raise ValueError("active_element_mask must contain at least one active element")

    total_dofs = node_array.shape[0] * 2
    force_array = np.asarray(force, dtype=float)
    if force_array.shape != (total_dofs,):
        raise ValueError(f"force shape must be {(total_dofs,)}")
    fixed = []
    for dof in np.asarray(fixed_dofs, dtype=int).ravel():
        if dof < 0 or dof >= total_dofs:
            raise ValueError("fixed_dofs contains an out-of-range degree of freedom")
        fixed.append(int(dof))
    fixed_dofs = np.asarray(sorted(set(fixed)), dtype=int)
    all_dofs = np.arange(total_dofs, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=True)

    return Mesh(
        nelx=nelx,
        nely=nely,
        length=length,
        height=height,
        nodes=node_array,
        elements=np.asarray(elements, dtype=int),
        element_dofs=np.asarray(element_dofs, dtype=int),
        centers=np.asarray(centers, dtype=float),
        fixed_dofs=fixed_dofs,
        free_dofs=free_dofs,
        force=force_array,
        element_area=dx * dy,
        element_grid_indices=np.asarray(element_grid_indices, dtype=int),
        active_element_mask=active_mask.copy(),
    )


def build_cantilever_mesh(
    nelx: int,
    nely: int,
    length: float = 3.0,
    height: float = 1.0,
) -> Mesh:
    _validate_geometry(nelx, nely, length, height)
    total_dofs = (nelx + 1) * (nely + 1) * 2
    fixed = []
    for iy in range(nely + 1):
        node = _node_id(0, iy, nelx)
        fixed.extend([2 * node, 2 * node + 1])

    force = np.zeros(total_dofs, dtype=float)
    midpoint_iy = nely // 2
    if nely % 2 == 0:
        load_node = _node_id(nelx, midpoint_iy, nelx)
        force[2 * load_node + 1] = -1.0
    else:
        # Split the unit load across the two right-edge nodes bracketing the midpoint.
        lower_load_node = _node_id(nelx, midpoint_iy, nelx)
        upper_load_node = _node_id(nelx, midpoint_iy + 1, nelx)
        force[2 * lower_load_node + 1] = -0.5
        force[2 * upper_load_node + 1] = -0.5

    return _build_structured_mesh(
        nelx=nelx,
        nely=nely,
        length=length,
        height=height,
        active_element_mask=np.ones((nely, nelx), dtype=bool),
        fixed_dofs=fixed,
        force=force,
    )


def build_mbb_mesh(
    nelx: int,
    nely: int,
    length: float = 3.0,
    height: float = 1.0,
) -> Mesh:
    _validate_geometry(nelx, nely, length, height)
    total_dofs = (nelx + 1) * (nely + 1) * 2

    fixed = []
    for iy in range(nely + 1):
        node = _node_id(0, iy, nelx)
        fixed.append(2 * node)
    right_bottom = _node_id(nelx, 0, nelx)
    fixed.append(2 * right_bottom + 1)

    force = np.zeros(total_dofs, dtype=float)
    top_left = _node_id(0, nely, nelx)
    force[2 * top_left + 1] = -1.0

    return _build_structured_mesh(
        nelx=nelx,
        nely=nely,
        length=length,
        height=height,
        active_element_mask=np.ones((nely, nelx), dtype=bool),
        fixed_dofs=fixed,
        force=force,
    )


def build_lbracket_mesh(
    nelx: int,
    nely: int,
    length: float = 2.0,
    height: float = 2.0,
) -> Mesh:
    _validate_geometry(nelx, nely, length, height)
    dx = length / nelx
    dy = height / nely
    total_dofs = (nelx + 1) * (nely + 1) * 2

    active_mask = np.zeros((nely, nelx), dtype=bool)
    for ey in range(nely):
        center_y = (ey + 0.5) * dy
        for ex in range(nelx):
            center_x = (ex + 0.5) * dx
            active_mask[ey, ex] = center_x <= 0.5 * length or center_y <= 0.5 * height

    active_nodes: set[int] = set()
    for ey in range(nely):
        for ex in range(nelx):
            if active_mask[ey, ex]:
                active_nodes.update(_active_element_nodes(ex, ey, nelx))

    fixed = []
    support_y = 0.5 * height
    tolerance = 10.0 * np.finfo(float).eps * max(1.0, height)
    for iy in range(nely + 1):
        y = iy * dy
        if y >= support_y - tolerance:
            node = _node_id(0, iy, nelx)
            fixed.extend([2 * node, 2 * node + 1])

    for node in range((nelx + 1) * (nely + 1)):
        if node not in active_nodes:
            fixed.extend([2 * node, 2 * node + 1])

    force = np.zeros(total_dofs, dtype=float)
    bottom_right = _node_id(nelx, 0, nelx)
    force[2 * bottom_right + 1] = -1.0

    return _build_structured_mesh(
        nelx=nelx,
        nely=nely,
        length=length,
        height=height,
        active_element_mask=active_mask,
        fixed_dofs=fixed,
        force=force,
    )
