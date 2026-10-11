from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from tpo.element import q4_plane_stress_stiffness
from tpo.mesh import Mesh
from tpo.runtime import RESULTS_ROOT
from tpo.simp_model import build_benchmark_mesh


BONMIN_EXECUTABLE = Path(r"C:\Users\JQ\AppData\Local\idaes\bin\bonmin.exe")
COUENNE_EXECUTABLE = Path(r"C:\Users\JQ\AppData\Local\idaes\bin\couenne.exe")
GDPHullEquilibriumTerm = tuple[int, int, float]
ElementArc = tuple[int, int]


@dataclass(frozen=True)
class GDPHullBinaryConfig:
    benchmark: str = "cantilever"
    nelx: int = 30
    nely: int = 10
    length: float = 3.0
    height: float = 1.0
    volfrac: float = 0.4
    displacement_bound: float = 1e4
    e0: float = 1.0
    nu: float = 0.3
    solver_name: str = "scip_direct"
    solver_time_limit: float = 300.0
    mip_gap: float = 1e-3
    tee: bool = False
    use_node_activation: bool = True
    use_element_connectivity: bool = False

    def __post_init__(self) -> None:
        label = str(self.benchmark).strip().lower().replace("-", "_")
        aliases = {
            "cantilever": "cantilever",
            "mbb": "mbb",
            "lbracket": "lbracket",
            "l_bracket": "lbracket",
            "lbeam": "lbracket",
            "l_beam": "lbracket",
        }
        object.__setattr__(self, "benchmark", aliases.get(label, label))


@dataclass(frozen=True)
class GDPHullBinaryModelData:
    mesh: Mesh
    element_ke: np.ndarray
    config: GDPHullBinaryConfig
    free_dof_to_index: dict[int, int]
    node_adjacent_elements: list[list[int]]
    equilibrium_terms: list[list[GDPHullEquilibriumTerm]]
    element_directed_edges: list[ElementArc]
    support_elements: list[int]
    load_elements: list[int]


@dataclass
class GDPHullBinarySolution:
    binary_density: np.ndarray
    free_displacement: np.ndarray
    full_displacement: np.ndarray
    compliance: float
    volume_fraction: float
    solid_element_count: int
    solver_status: str
    termination_condition: str
    elapsed_seconds: float
    best_bound: float | None
    mip_gap: float | None
    displacement_bound_saturation: float
    model_data: GDPHullBinaryModelData
    stability_cut_count: int = 0
    stability_rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class GDPHullStabilityCheck:
    accepted: bool
    reason: str
    displacement_bound_saturation: float
    linear_response_compliance: float | None = None
    linear_response_max_displacement: float | None = None
    linear_response_bound_saturation: float | None = None
    linear_response_full_displacement: np.ndarray | None = None


def _validate_config(config: GDPHullBinaryConfig) -> None:
    if config.nelx <= 0 or config.nely <= 0:
        raise ValueError("nelx and nely must be positive")
    if not np.isfinite(config.length) or config.length <= 0.0:
        raise ValueError("length must be positive and finite")
    if not np.isfinite(config.height) or config.height <= 0.0:
        raise ValueError("height must be positive and finite")
    if not np.isfinite(config.volfrac) or not (0.0 < config.volfrac <= 1.0):
        raise ValueError("volfrac must be finite and satisfy 0 < volfrac <= 1")
    if not np.isfinite(config.displacement_bound) or config.displacement_bound <= 0.0:
        raise ValueError("displacement_bound must be positive and finite")
    if not np.isfinite(config.e0) or config.e0 <= 0.0:
        raise ValueError("e0 must be positive and finite")
    if not np.isfinite(config.solver_time_limit) or config.solver_time_limit <= 0.0:
        raise ValueError("solver_time_limit must be positive and finite")
    if not np.isfinite(config.mip_gap) or config.mip_gap < 0.0:
        raise ValueError("mip_gap must be nonnegative and finite")


def _node_adjacent_elements(mesh: Mesh) -> list[list[int]]:
    adjacent: list[list[int]] = [[] for _ in range(mesh.nodes.shape[0])]
    for element_index, nodes in enumerate(mesh.elements):
        for node in nodes:
            adjacent[int(node)].append(int(element_index))
    return adjacent


def _free_dof_to_index(mesh: Mesh) -> dict[int, int]:
    return {int(global_dof): index for index, global_dof in enumerate(mesh.free_dofs)}


def _element_directed_edges(mesh: Mesh) -> list[ElementArc]:
    local_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    edge_to_elements: dict[tuple[int, int], list[int]] = {}
    for element_index, nodes in enumerate(mesh.elements):
        for a, b in local_edges:
            edge = tuple(sorted((int(nodes[a]), int(nodes[b]))))
            edge_to_elements.setdefault(edge, []).append(int(element_index))

    directed_edges: set[ElementArc] = set()
    for element_indices in edge_to_elements.values():
        if len(element_indices) < 2:
            continue
        for source in element_indices:
            for target in element_indices:
                if source != target:
                    directed_edges.add((source, target))
    return sorted(directed_edges)


def _elements_touching_nodes(mesh: Mesh, node_indices: np.ndarray) -> list[int]:
    node_set = {int(node) for node in np.asarray(node_indices, dtype=int).ravel()}
    if not node_set:
        return []
    return [
        int(element_index)
        for element_index, nodes in enumerate(mesh.elements)
        if any(int(node) in node_set for node in nodes)
    ]


def _support_elements(mesh: Mesh) -> list[int]:
    fixed_nodes = np.unique(np.asarray(mesh.fixed_dofs, dtype=int) // 2)
    return _elements_touching_nodes(mesh, fixed_nodes)


def _load_elements(mesh: Mesh) -> list[int]:
    loaded_dofs = np.flatnonzero(np.abs(mesh.force) > 0.0)
    loaded_nodes = np.unique(loaded_dofs // 2)
    return _elements_touching_nodes(mesh, loaded_nodes)


def _build_solid_phase_equilibrium_terms(
    mesh: Mesh,
    element_ke: np.ndarray,
    free_dof_to_index: dict[int, int],
) -> list[list[GDPHullEquilibriumTerm]]:
    rows: list[list[GDPHullEquilibriumTerm]] = [[] for _ in range(mesh.free_dofs.size)]
    for element_index, dofs in enumerate(mesh.element_dofs):
        for local_row, global_row in enumerate(dofs):
            row_index = free_dof_to_index.get(int(global_row))
            if row_index is None:
                continue
            for local_col, global_col in enumerate(dofs):
                if int(global_col) not in free_dof_to_index:
                    continue
                value = float(element_ke[int(local_row), int(local_col)])
                if value != 0.0:
                    rows[row_index].append((int(element_index), int(local_col), value))
    return rows


def _global_displacement_expression(
    model: pyo.ConcreteModel,
    data: GDPHullBinaryModelData,
    global_dof: int,
):
    free_index = data.free_dof_to_index.get(int(global_dof))
    if free_index is None:
        return 0.0
    return model.u[free_index]


def _build_data(config: GDPHullBinaryConfig) -> GDPHullBinaryModelData:
    mesh = build_benchmark_mesh(config)
    element_ke = q4_plane_stress_stiffness(
        width=config.length / config.nelx,
        height=config.height / config.nely,
        young=config.e0,
        nu=config.nu,
    )
    free_dof_to_index = _free_dof_to_index(mesh)
    return GDPHullBinaryModelData(
        mesh=mesh,
        element_ke=element_ke,
        config=config,
        free_dof_to_index=free_dof_to_index,
        node_adjacent_elements=_node_adjacent_elements(mesh),
        equilibrium_terms=_build_solid_phase_equilibrium_terms(
            mesh,
            element_ke,
            free_dof_to_index,
        ),
        element_directed_edges=_element_directed_edges(mesh),
        support_elements=_support_elements(mesh),
        load_elements=_load_elements(mesh),
    )


def _add_element_connectivity_constraints(
    model: pyo.ConcreteModel,
    data: GDPHullBinaryModelData,
) -> None:
    mesh = data.mesh
    if not data.support_elements:
        raise ValueError("element connectivity requires at least one support-adjacent element")
    if not data.load_elements:
        raise ValueError("element connectivity requires at least one load-adjacent element")

    edge_in: list[list[ElementArc]] = [[] for _ in range(mesh.element_count)]
    edge_out: list[list[ElementArc]] = [[] for _ in range(mesh.element_count)]
    for source, target in data.element_directed_edges:
        edge_out[source].append((source, target))
        edge_in[target].append((source, target))

    big_m = float(mesh.element_count)
    model.CA = pyo.Set(initialize=data.element_directed_edges, dimen=2)
    model.CR = pyo.Set(initialize=data.support_elements)
    model.connectivity_flow = pyo.Var(model.CA, domain=pyo.NonNegativeReals, bounds=(0.0, big_m))
    model.connectivity_source_flow = pyo.Var(
        model.CR,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, big_m),
    )

    def arc_tail_capacity_rule(m: pyo.ConcreteModel, source: int, target: int):
        return m.connectivity_flow[source, target] <= big_m * m.y[source]

    def arc_head_capacity_rule(m: pyo.ConcreteModel, source: int, target: int):
        return m.connectivity_flow[source, target] <= big_m * m.y[target]

    def source_capacity_rule(m: pyo.ConcreteModel, element: int):
        return m.connectivity_source_flow[element] <= big_m * m.y[element]

    def flow_balance_rule(m: pyo.ConcreteModel, element: int):
        source_flow = (
            m.connectivity_source_flow[element]
            if element in data.support_elements
            else 0.0
        )
        incoming = sum(m.connectivity_flow[source, target] for source, target in edge_in[int(element)])
        outgoing = sum(m.connectivity_flow[source, target] for source, target in edge_out[int(element)])
        return source_flow + incoming - outgoing >= m.y[element]

    model.connectivity_arc_tail_capacity = pyo.Constraint(model.CA, rule=arc_tail_capacity_rule)
    model.connectivity_arc_head_capacity = pyo.Constraint(model.CA, rule=arc_head_capacity_rule)
    model.connectivity_source_capacity = pyo.Constraint(model.CR, rule=source_capacity_rule)
    model.connectivity_flow_balance = pyo.Constraint(model.E, rule=flow_balance_rule)
    model.load_connected = pyo.Constraint(
        expr=sum(model.y[e] for e in data.load_elements) >= 1.0
    )


def build_gdp_hull_binary_model(
    config: GDPHullBinaryConfig,
) -> tuple[pyo.ConcreteModel, GDPHullBinaryModelData]:
    _validate_config(config)
    data = _build_data(config)
    mesh = data.mesh
    bound = float(config.displacement_bound)

    model = pyo.ConcreteModel()
    model.E = pyo.RangeSet(0, mesh.element_count - 1)
    model.A = pyo.RangeSet(0, 7)
    model.F = pyo.RangeSet(0, mesh.free_dofs.size - 1)
    model.N = pyo.RangeSet(0, mesh.nodes.shape[0] - 1)

    model.y = pyo.Var(model.E, domain=pyo.Binary, initialize=1.0)
    model.z = pyo.Var(model.N, domain=pyo.Binary, initialize=1.0)
    model.u = pyo.Var(model.F, bounds=(-bound, bound), initialize=0.0)
    model.us = pyo.Var(model.E, model.A, bounds=(-bound, bound), initialize=0.0)

    def solid_lower_rule(m: pyo.ConcreteModel, e: int, a: int):
        return -bound * m.y[e] <= m.us[e, a]

    def solid_upper_rule(m: pyo.ConcreteModel, e: int, a: int):
        return m.us[e, a] <= bound * m.y[e]

    def void_lower_rule(m: pyo.ConcreteModel, e: int, a: int):
        global_dof = int(mesh.element_dofs[int(e), int(a)])
        displacement = _global_displacement_expression(m, data, global_dof)
        return -bound * (1.0 - m.y[e]) <= displacement - m.us[e, a]

    def void_upper_rule(m: pyo.ConcreteModel, e: int, a: int):
        global_dof = int(mesh.element_dofs[int(e), int(a)])
        displacement = _global_displacement_expression(m, data, global_dof)
        return displacement - m.us[e, a] <= bound * (1.0 - m.y[e])

    model.solid_lower = pyo.Constraint(model.E, model.A, rule=solid_lower_rule)
    model.solid_upper = pyo.Constraint(model.E, model.A, rule=solid_upper_rule)
    model.void_lower = pyo.Constraint(model.E, model.A, rule=void_lower_rule)
    model.void_upper = pyo.Constraint(model.E, model.A, rule=void_upper_rule)

    if config.use_node_activation:
        element_node_pairs = [
            (int(element_index), int(node))
            for element_index, nodes in enumerate(mesh.elements)
            for node in nodes
        ]
        model.EN = pyo.Set(initialize=element_node_pairs, dimen=2)

        def node_active_lower_rule(m: pyo.ConcreteModel, e: int, node: int):
            return m.z[node] >= m.y[e]

        def node_active_upper_rule(m: pyo.ConcreteModel, node: int):
            adjacent = data.node_adjacent_elements[int(node)]
            if not adjacent:
                return m.z[node] == 0
            return m.z[node] <= sum(m.y[e] for e in adjacent)

        def free_displacement_node_upper_rule(m: pyo.ConcreteModel, i: int):
            node = int(mesh.free_dofs[int(i)]) // 2
            return m.u[i] <= bound * m.z[node]

        def free_displacement_node_lower_rule(m: pyo.ConcreteModel, i: int):
            node = int(mesh.free_dofs[int(i)]) // 2
            return -bound * m.z[node] <= m.u[i]

        model.node_active_lower = pyo.Constraint(model.EN, rule=node_active_lower_rule)
        model.node_active_upper = pyo.Constraint(model.N, rule=node_active_upper_rule)
        model.free_node_upper = pyo.Constraint(model.F, rule=free_displacement_node_upper_rule)
        model.free_node_lower = pyo.Constraint(model.F, rule=free_displacement_node_lower_rule)

    if config.use_element_connectivity:
        _add_element_connectivity_constraints(model, data)

    def equilibrium_rule(m: pyo.ConcreteModel, free_row: int):
        total = sum(
            value * m.us[element_index, local_col]
            for element_index, local_col, value in data.equilibrium_terms[int(free_row)]
        )
        return total == float(mesh.force[mesh.free_dofs[int(free_row)]])

    model.equilibrium = pyo.Constraint(model.F, rule=equilibrium_rule)
    model.volume_constraint = pyo.Constraint(
        expr=sum(model.y[e] * mesh.element_area for e in model.E)
        <= config.volfrac * mesh.element_count * mesh.element_area
    )
    model.compliance = pyo.Expression(
        expr=sum(float(mesh.force[int(global_dof)]) * model.u[i] for i, global_dof in enumerate(mesh.free_dofs))
    )
    model.objective = pyo.Objective(expr=model.compliance, sense=pyo.minimize)
    return model, data


def _create_solver(config: GDPHullBinaryConfig) -> Any:
    name = str(config.solver_name)
    if name == "bonmin":
        solver = pyo.SolverFactory(name, executable=str(BONMIN_EXECUTABLE))
        solver.options["bonmin.time_limit"] = float(config.solver_time_limit)
        solver.options["max_cpu_time"] = float(config.solver_time_limit)
        return solver
    if name == "couenne":
        solver = pyo.SolverFactory(name, executable=str(COUENNE_EXECUTABLE))
        solver.options["bonmin.time_limit"] = float(config.solver_time_limit)
        solver.options["max_cpu_time"] = float(config.solver_time_limit)
        return solver
    solver = pyo.SolverFactory(name)
    if name == "scip_direct":
        solver.options["limits/time"] = float(config.solver_time_limit)
        solver.options["limits/gap"] = float(config.mip_gap)
    return solver


def _read_variable_vector(indexed_var: pyo.Var, indices, expected_size: int, name: str) -> np.ndarray:
    values = np.zeros(expected_size, dtype=float)
    for position, index in enumerate(indices):
        raw = pyo.value(indexed_var[index], exception=False)
        if raw is None:
            raise RuntimeError(f"{name}[{index}] has no value")
        value = float(raw)
        if not np.isfinite(value):
            raise RuntimeError(f"{name}[{index}] must be finite")
        values[position] = value
    return values


def _solver_best_bound(results) -> float | None:
    for container_name in ("problem", "solver"):
        container = getattr(results, container_name, None)
        if container is None:
            continue
        for attribute in ("lower_bound", "lower bound", "Lower bound"):
            try:
                value = getattr(container, attribute)
            except (AttributeError, TypeError):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _solution_mip_gap(compliance: float, best_bound: float | None) -> float | None:
    if best_bound is None or not np.isfinite(best_bound):
        return None
    denominator = max(abs(compliance), 1e-12)
    return max(0.0, float((compliance - best_bound) / denominator))


def extract_gdp_hull_binary_solution(
    model: pyo.ConcreteModel,
    data: GDPHullBinaryModelData,
    solver_results,
    elapsed_seconds: float,
) -> GDPHullBinarySolution:
    mesh = data.mesh
    y_values = _read_variable_vector(model.y, model.E, mesh.element_count, "binary density")
    binary_density = np.rint(np.clip(y_values, 0.0, 1.0)).astype(float)
    free_displacement = _read_variable_vector(
        model.u,
        model.F,
        mesh.free_dofs.size,
        "free displacement",
    )
    full_displacement = np.zeros(mesh.total_dofs, dtype=float)
    full_displacement[mesh.free_dofs] = free_displacement
    compliance = float(mesh.force @ full_displacement)
    volume_fraction = float(np.mean(binary_density))
    best_bound = _solver_best_bound(solver_results)
    return GDPHullBinarySolution(
        binary_density=binary_density,
        free_displacement=free_displacement,
        full_displacement=full_displacement,
        compliance=compliance,
        volume_fraction=volume_fraction,
        solid_element_count=int(np.count_nonzero(binary_density)),
        solver_status=str(solver_results.solver.status),
        termination_condition=str(solver_results.solver.termination_condition),
        elapsed_seconds=float(elapsed_seconds),
        best_bound=best_bound,
        mip_gap=_solution_mip_gap(compliance, best_bound),
        displacement_bound_saturation=float(
            np.max(np.abs(full_displacement)) / data.config.displacement_bound
        ),
        model_data=data,
    )


def _volume_infeasibility_reason(solution: GDPHullBinarySolution, tolerance: float = 1e-9) -> str | None:
    limit = float(solution.model_data.config.volfrac)
    if solution.volume_fraction <= limit + tolerance:
        return None
    return (
        f"volume fraction {solution.volume_fraction:.6g} exceeds "
        f"limit {limit:.6g}"
    )


def _with_linear_response_metrics(
    solution: GDPHullBinarySolution,
    stability: GDPHullStabilityCheck,
) -> GDPHullBinarySolution:
    if stability.linear_response_full_displacement is None:
        raise ValueError("accepted stability check must include linear response displacement")
    full_displacement = np.asarray(stability.linear_response_full_displacement, dtype=float).copy()
    free_displacement = full_displacement[solution.model_data.mesh.free_dofs].copy()
    compliance = float(stability.linear_response_compliance)
    max_displacement = float(stability.linear_response_max_displacement)
    return GDPHullBinarySolution(
        binary_density=solution.binary_density.copy(),
        free_displacement=free_displacement,
        full_displacement=full_displacement,
        compliance=compliance,
        volume_fraction=solution.volume_fraction,
        solid_element_count=solution.solid_element_count,
        solver_status=solution.solver_status,
        termination_condition=solution.termination_condition,
        elapsed_seconds=solution.elapsed_seconds,
        best_bound=solution.best_bound,
        mip_gap=solution.mip_gap,
        displacement_bound_saturation=max_displacement / solution.model_data.config.displacement_bound,
        model_data=solution.model_data,
        stability_cut_count=solution.stability_cut_count,
        stability_rejection_reasons=solution.stability_rejection_reasons,
    )


def _validate_fixed_binary_density(values, mesh: Mesh) -> np.ndarray:
    density = np.asarray(values, dtype=float)
    if density.shape != (mesh.element_count,):
        raise ValueError(f"fixed_binary_density shape must be {(mesh.element_count,)}")
    if not np.all(np.isfinite(density)):
        raise ValueError("fixed_binary_density must contain finite values")
    rounded = np.rint(density)
    if not np.all(np.isin(rounded, [0.0, 1.0])):
        raise ValueError("fixed_binary_density must contain only 0/1 values")
    return rounded.astype(float)


def add_no_good_binary_topology_cut(
    model: pyo.ConcreteModel,
    data: GDPHullBinaryModelData,
    binary_density,
) -> None:
    density = _validate_fixed_binary_density(binary_density, data.mesh)
    if not hasattr(model, "no_good_binary_topologies"):
        model.no_good_binary_topologies = pyo.ConstraintList()
    expr = sum(
        (1.0 - model.y[element_index]) if value > 0.5 else model.y[element_index]
        for element_index, value in enumerate(density)
    )
    model.no_good_binary_topologies.add(expr >= 1.0)


def _solve_active_binary_linear_response(
    data: GDPHullBinaryModelData,
    binary_density: np.ndarray,
) -> np.ndarray:
    mesh = data.mesh
    solid_elements = np.flatnonzero(binary_density > 0.5)
    if solid_elements.size == 0:
        raise RuntimeError("topology contains no solid elements")

    active_dofs = np.unique(mesh.element_dofs[solid_elements].ravel())
    loaded_dofs = np.flatnonzero(np.abs(mesh.force) > 0.0)
    inactive_loaded_dofs = np.setdiff1d(loaded_dofs, active_dofs, assume_unique=False)
    if inactive_loaded_dofs.size:
        raise RuntimeError("load dof is not attached to solid topology")

    active_free_dofs = np.setdiff1d(active_dofs, mesh.fixed_dofs, assume_unique=False)
    if active_free_dofs.size == 0:
        raise RuntimeError("topology has no active free displacement dofs")

    active_free_map = {int(global_dof): index for index, global_dof in enumerate(active_free_dofs)}
    row_indices = []
    col_indices = []
    values = []
    for element_index in solid_elements:
        for local_row, global_row in enumerate(mesh.element_dofs[element_index]):
            row_index = active_free_map.get(int(global_row))
            if row_index is None:
                continue
            for local_col, global_col in enumerate(mesh.element_dofs[element_index]):
                col_index = active_free_map.get(int(global_col))
                if col_index is None:
                    continue
                row_indices.append(row_index)
                col_indices.append(col_index)
                values.append(float(data.element_ke[local_row, local_col]))

    matrix = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(active_free_dofs.size, active_free_dofs.size),
    ).tocsr()
    rhs = mesh.force[active_free_dofs]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            active_free_displacement = np.asarray(spsolve(matrix, rhs), dtype=float)
    except MatrixRankWarning as exc:
        raise RuntimeError("stiffness matrix is singular") from exc
    if not np.all(np.isfinite(active_free_displacement)):
        raise RuntimeError("non-finite displacement values")

    full_displacement = np.zeros(mesh.total_dofs, dtype=float)
    full_displacement[active_free_dofs] = active_free_displacement
    return full_displacement


def check_binary_topology_stability(
    data: GDPHullBinaryModelData,
    binary_density,
    displacement_bound_saturation: float,
    saturation_limit: float = 0.95,
) -> GDPHullStabilityCheck:
    density = _validate_fixed_binary_density(binary_density, data.mesh)
    saturation = float(displacement_bound_saturation)
    if not np.isfinite(saturation):
        return GDPHullStabilityCheck(
            accepted=False,
            reason="displacement bound saturation is not finite",
            displacement_bound_saturation=saturation,
        )
    if not np.isfinite(saturation_limit) or saturation_limit <= 0.0:
        raise ValueError("saturation_limit must be positive and finite")
    if saturation >= float(saturation_limit):
        return GDPHullStabilityCheck(
            accepted=False,
            reason=f"displacement bound saturation {saturation:.6g} >= {float(saturation_limit):.6g}",
            displacement_bound_saturation=saturation,
        )

    try:
        full_displacement = _solve_active_binary_linear_response(data, density)
    except RuntimeError as exc:
        return GDPHullStabilityCheck(
            accepted=False,
            reason=f"linear response failed: {exc}",
            displacement_bound_saturation=saturation,
        )
    compliance = float(data.mesh.force @ full_displacement)
    max_displacement = float(np.max(np.abs(full_displacement)))
    linear_saturation = max_displacement / float(data.config.displacement_bound)
    if not np.isfinite(compliance) or not np.isfinite(max_displacement):
        return GDPHullStabilityCheck(
            accepted=False,
            reason="linear response returned non-finite values",
            displacement_bound_saturation=saturation,
            linear_response_compliance=compliance,
            linear_response_max_displacement=max_displacement,
            linear_response_bound_saturation=linear_saturation,
        )
    if compliance <= 0.0:
        return GDPHullStabilityCheck(
            accepted=False,
            reason=f"linear response returned nonphysical compliance {compliance:.6g}",
            displacement_bound_saturation=saturation,
            linear_response_compliance=compliance,
            linear_response_max_displacement=max_displacement,
            linear_response_bound_saturation=linear_saturation,
        )
    if linear_saturation >= float(saturation_limit):
        return GDPHullStabilityCheck(
            accepted=False,
            reason=(
                "linear response displacement bound saturation "
                f"{linear_saturation:.6g} >= {float(saturation_limit):.6g}"
            ),
            displacement_bound_saturation=saturation,
            linear_response_compliance=compliance,
            linear_response_max_displacement=max_displacement,
            linear_response_bound_saturation=linear_saturation,
        )
    return GDPHullStabilityCheck(
        accepted=True,
        reason="accepted",
        displacement_bound_saturation=saturation,
        linear_response_compliance=compliance,
        linear_response_max_displacement=max_displacement,
        linear_response_bound_saturation=linear_saturation,
        linear_response_full_displacement=full_displacement,
    )


def apply_initial_binary_density(
    model: pyo.ConcreteModel,
    data: GDPHullBinaryModelData,
    initial_binary_density,
) -> np.ndarray:
    initial_density = _validate_fixed_binary_density(initial_binary_density, data.mesh)
    for element_index, value in enumerate(initial_density):
        model.y[element_index].set_value(float(value))

    if hasattr(model, "z"):
        for node_index, adjacent in enumerate(data.node_adjacent_elements):
            active = any(initial_density[element_index] > 0.5 for element_index in adjacent)
            model.z[node_index].set_value(1.0 if active else 0.0)
    return initial_density


def solve_gdp_hull_binary(
    config: GDPHullBinaryConfig,
    fixed_binary_density: np.ndarray | None = None,
    initial_binary_density: np.ndarray | None = None,
) -> GDPHullBinarySolution:
    model, data = build_gdp_hull_binary_model(config)
    if initial_binary_density is not None:
        apply_initial_binary_density(model, data, initial_binary_density)
    if fixed_binary_density is not None:
        fixed_density = _validate_fixed_binary_density(fixed_binary_density, data.mesh)
        for element_index, value in enumerate(fixed_density):
            model.y[element_index].fix(float(value))

    solver = _create_solver(config)
    if not solver.available(False):
        raise RuntimeError(f"solver is not available: {config.solver_name}")
    started_at = time.perf_counter()
    results = solver.solve(model, tee=config.tee)
    elapsed = time.perf_counter() - started_at
    termination = results.solver.termination_condition
    acceptable = {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
    }
    if termination not in acceptable:
        raise RuntimeError(
            "GDP-Hull binary solve failed: "
            f"status={results.solver.status}, termination={termination}"
        )
    return extract_gdp_hull_binary_solution(model, data, results, elapsed)


def solve_gdp_hull_binary_with_stability_cuts(
    config: GDPHullBinaryConfig,
    initial_binary_density: np.ndarray | None = None,
    max_no_good_cuts: int = 10,
    saturation_limit: float = 0.95,
) -> GDPHullBinarySolution:
    if max_no_good_cuts < 0:
        raise ValueError("max_no_good_cuts must be nonnegative")
    model, data = build_gdp_hull_binary_model(config)
    if initial_binary_density is not None:
        apply_initial_binary_density(model, data, initial_binary_density)

    total_elapsed = 0.0
    rejection_reasons: list[str] = []
    for attempt_index in range(int(max_no_good_cuts) + 1):
        solver = _create_solver(config)
        if not solver.available(False):
            raise RuntimeError(f"solver is not available: {config.solver_name}")
        started_at = time.perf_counter()
        results = solver.solve(model, tee=config.tee)
        elapsed = time.perf_counter() - started_at
        total_elapsed += elapsed
        termination = results.solver.termination_condition
        acceptable = {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.maxIterations,
        }
        if termination not in acceptable:
            raise RuntimeError(
                "GDP-Hull binary solve failed during stability screening: "
                f"status={results.solver.status}, termination={termination}"
            )
        solution = extract_gdp_hull_binary_solution(model, data, results, elapsed)
        solution.elapsed_seconds = total_elapsed
        volume_reason = _volume_infeasibility_reason(solution)
        if volume_reason is not None:
            raise RuntimeError(
                "GDP-Hull binary solver did not return a volume-feasible incumbent: "
                f"{volume_reason}; status={results.solver.status}, termination={termination}"
            )
        stability = check_binary_topology_stability(
            data,
            solution.binary_density,
            solution.displacement_bound_saturation,
            saturation_limit=saturation_limit,
        )
        if stability.accepted:
            solution = _with_linear_response_metrics(solution, stability)
            solution.stability_cut_count = attempt_index
            solution.stability_rejection_reasons = tuple(rejection_reasons)
            return solution
        rejection_reasons.append(stability.reason)
        if config.tee:
            print(f"Rejected topology {attempt_index + 1}: {stability.reason}")
        if attempt_index >= int(max_no_good_cuts):
            raise RuntimeError(
                "GDP-Hull stability screening failed: "
                f"rejected {attempt_index + 1} topology/topologies; last reason: {stability.reason}"
            )
        add_no_good_binary_topology_cut(model, data, solution.binary_density)

    raise RuntimeError("GDP-Hull stability screening failed unexpectedly")


def default_displacement_bound(
    benchmark: str,
    nelx: int,
    nely: int,
    factor: float = 10.0,
    fallback: float = 1e4,
) -> float:
    result_dir = RESULTS_ROOT / f"oc_simp_{benchmark}_{nelx}x{nely}"
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        import json

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        value = float(summary.get("max_displacement", fallback))
        if np.isfinite(value) and value > 0.0:
            return float(factor * value)
    return float(fallback)


__all__ = [
    "GDPHullBinaryConfig",
    "GDPHullBinaryModelData",
    "GDPHullBinarySolution",
    "GDPHullStabilityCheck",
    "add_no_good_binary_topology_cut",
    "apply_initial_binary_density",
    "build_gdp_hull_binary_model",
    "check_binary_topology_stability",
    "default_displacement_bound",
    "extract_gdp_hull_binary_solution",
    "solve_gdp_hull_binary",
    "solve_gdp_hull_binary_with_stability_cuts",
]
