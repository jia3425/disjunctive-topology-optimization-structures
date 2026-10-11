from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

from tpo.element import q4_plane_stress_stiffness
from tpo.fea import solve_linear_response
from tpo.filtering import DensityFilter, build_density_filter
from tpo.gdp_hull_binary import (
    GDPHullBinaryConfig,
    _build_solid_phase_equilibrium_terms,
    _element_directed_edges,
    _free_dof_to_index,
    _global_displacement_expression,
    _load_elements,
    _solve_active_binary_linear_response,
    _support_elements,
)
from tpo.mesh import Mesh
from tpo.runtime import SolverSettings, create_ipopt_solver
from tpo.simp_model import build_benchmark_mesh


@dataclass(frozen=True)
class GDPHullRelaxationConfig:
    benchmark: str = "cantilever"
    nelx: int = 30
    nely: int = 10
    length: float = 3.0
    height: float = 1.0
    volfrac: float = 0.4
    displacement_bound: float = 1e4
    e0: float = 1.0
    nu: float = 0.3
    ipopt_tol: float = 1e-7
    ipopt_max_iter: int = 1200
    tee: bool = False
    use_element_connectivity: bool = False
    fix_load_elements: bool = False
    phase_filter_radius: float | None = None
    heaviside_eta: float = 0.5
    use_tight_displacement_bounds: bool = False
    tight_displacement_bound_safety_factor: float = 20.0
    tight_displacement_bound_floor_fraction: float = 1e-3
    enforce_full_solid_compliance_lower_bound: bool = False

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
class GDPHullRelaxationStage:
    penalty: float = 0.0
    grayness_limit: float | None = None
    heaviside_beta: float | None = None
    solid_displacement_coupling_penalty: float = 0.0
    solid_displacement_coupling_phase_law: str = "linear"
    solid_displacement_coupling_phase_exponent: float = 1.0
    solid_displacement_coupling_ramp_q: float = 8.0
    solid_displacement_coupling_norm: str = "bound"
    solid_displacement_coupling_mode: str = "penalty"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "solid_displacement_coupling_phase_law",
            str(self.solid_displacement_coupling_phase_law).strip().lower(),
        )
        object.__setattr__(
            self,
            "solid_displacement_coupling_norm",
            str(self.solid_displacement_coupling_norm).strip().lower(),
        )
        object.__setattr__(
            self,
            "solid_displacement_coupling_mode",
            str(self.solid_displacement_coupling_mode).strip().lower(),
        )


@dataclass(frozen=True)
class GDPHullRelaxationModelData:
    mesh: Mesh
    element_ke: np.ndarray
    config: GDPHullRelaxationConfig
    free_dof_to_index: dict[int, int]
    equilibrium_terms: list[list[tuple[int, int, float]]]
    reference_compliance: float
    element_directed_edges: list[tuple[int, int]]
    support_elements: list[int]
    load_elements: list[int]
    full_solid_compliance: float
    full_solid_displacement: np.ndarray
    displacement_bounds: np.ndarray
    phase_filter: DensityFilter | None = None


@dataclass(frozen=True)
class GDPHullGraynessMetrics:
    global_grayness: float
    max_element_grayness: float
    gray_element_count: int
    mean_solid_fraction: float


@dataclass
class GDPHullRelaxationStageResult:
    stage_index: int
    penalty: float
    grayness_limit: float | None
    heaviside_beta: float | None
    compliance: float
    normalized_objective: float
    global_grayness: float
    max_element_grayness: float
    volume_fraction: float
    solid_displacement_coupling_penalty: float
    solid_displacement_coupling: float
    solid_displacement_coupling_phase_law: str
    solid_displacement_coupling_phase_exponent: float
    solid_displacement_coupling_ramp_q: float
    solid_displacement_coupling_norm: str
    solid_displacement_coupling_mode: str
    solver_status: str
    termination_condition: str
    elapsed_seconds: float


@dataclass
class GDPHullRelaxationSolution:
    ys: np.ndarray
    yv: np.ndarray
    free_displacement: np.ndarray
    full_displacement: np.ndarray
    solid_displacement_components: np.ndarray
    continuous_compliance: float
    volume_fraction: float
    global_grayness: float
    max_element_grayness: float
    stage_results: list[GDPHullRelaxationStageResult]
    rounded_binary_density: np.ndarray
    rounded_binary_compliance: float | None
    rounded_binary_volume_fraction: float
    rounded_binary_displacement: np.ndarray | None
    rounded_response_status: str
    model_data: GDPHullRelaxationModelData
    completed_all_stages: bool = True
    failed_stage_index: int | None = None
    failed_stage_reason: str | None = None
    raw_ys: np.ndarray | None = None
    raw_yv: np.ndarray | None = None
    filtered_ys: np.ndarray | None = None


def _validate_config(config: GDPHullRelaxationConfig) -> None:
    if config.nelx <= 0 or config.nely <= 0:
        raise ValueError("nelx and nely must be positive")
    if not np.isfinite(config.length) or config.length <= 0.0:
        raise ValueError("length must be positive and finite")
    if not np.isfinite(config.height) or config.height <= 0.0:
        raise ValueError("height must be positive and finite")
    if not np.isfinite(config.volfrac) or not (0.0 < config.volfrac <= 1.0):
        raise ValueError("volfrac must satisfy 0 < volfrac <= 1")
    if not np.isfinite(config.displacement_bound) or config.displacement_bound <= 0.0:
        raise ValueError("displacement_bound must be positive and finite")
    if not np.isfinite(config.e0) or config.e0 <= 0.0:
        raise ValueError("e0 must be positive and finite")
    if not np.isfinite(config.ipopt_tol) or config.ipopt_tol <= 0.0:
        raise ValueError("ipopt_tol must be positive and finite")
    if int(config.ipopt_max_iter) <= 0:
        raise ValueError("ipopt_max_iter must be positive")
    if config.phase_filter_radius is not None:
        if not np.isfinite(config.phase_filter_radius) or config.phase_filter_radius <= 0.0:
            raise ValueError("phase_filter_radius must be positive and finite")
    if not np.isfinite(config.heaviside_eta) or not (0.0 < config.heaviside_eta < 1.0):
        raise ValueError("heaviside_eta must satisfy 0 < eta < 1")
    if not np.isfinite(config.tight_displacement_bound_safety_factor):
        raise ValueError("tight_displacement_bound_safety_factor must be finite")
    if config.tight_displacement_bound_safety_factor <= 0.0:
        raise ValueError("tight_displacement_bound_safety_factor must be positive")
    if not np.isfinite(config.tight_displacement_bound_floor_fraction):
        raise ValueError("tight_displacement_bound_floor_fraction must be finite")
    if config.tight_displacement_bound_floor_fraction <= 0.0:
        raise ValueError("tight_displacement_bound_floor_fraction must be positive")


def _validate_stage(stage: GDPHullRelaxationStage) -> None:
    if not np.isfinite(stage.penalty) or stage.penalty < 0.0:
        raise ValueError("stage penalty must be nonnegative and finite")
    if stage.grayness_limit is not None:
        if not np.isfinite(stage.grayness_limit) or not (0.0 <= stage.grayness_limit <= 1.0):
            raise ValueError("grayness_limit must satisfy 0 <= value <= 1")
    if stage.heaviside_beta is not None:
        if not np.isfinite(stage.heaviside_beta) or stage.heaviside_beta <= 0.0:
            raise ValueError("heaviside_beta must be positive and finite")
    if not np.isfinite(stage.solid_displacement_coupling_penalty):
        raise ValueError("solid_displacement_coupling_penalty must be finite")
    if stage.solid_displacement_coupling_penalty < 0.0:
        raise ValueError("solid_displacement_coupling_penalty must be nonnegative")
    if stage.solid_displacement_coupling_phase_law not in {"linear", "simp", "ramp"}:
        raise ValueError("solid_displacement_coupling_phase_law must be linear, simp, or ramp")
    if not np.isfinite(stage.solid_displacement_coupling_phase_exponent):
        raise ValueError("solid_displacement_coupling_phase_exponent must be finite")
    if stage.solid_displacement_coupling_phase_exponent <= 0.0:
        raise ValueError("solid_displacement_coupling_phase_exponent must be positive")
    if not np.isfinite(stage.solid_displacement_coupling_ramp_q):
        raise ValueError("solid_displacement_coupling_ramp_q must be finite")
    if stage.solid_displacement_coupling_ramp_q < 0.0:
        raise ValueError("solid_displacement_coupling_ramp_q must be nonnegative")
    if stage.solid_displacement_coupling_norm not in {"bound", "energy"}:
        raise ValueError("solid_displacement_coupling_norm must be bound or energy")
    if stage.solid_displacement_coupling_mode not in {"penalty", "equality"}:
        raise ValueError("solid_displacement_coupling_mode must be penalty or equality")


def heaviside_projection(values, beta: float, eta: float = 0.5) -> np.ndarray:
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    if not np.isfinite(eta) or not (0.0 < eta < 1.0):
        raise ValueError("eta must satisfy 0 < eta < 1")
    vector = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must contain only finite entries")
    denominator = np.tanh(float(beta) * float(eta)) + np.tanh(float(beta) * (1.0 - float(eta)))
    projected = (
        np.tanh(float(beta) * float(eta))
        + np.tanh(float(beta) * (vector - float(eta)))
    ) / denominator
    return np.clip(projected, 0.0, 1.0)


def grayness_metrics(ys, element_areas) -> GDPHullGraynessMetrics:
    density = np.asarray(ys, dtype=float)
    areas = np.asarray(element_areas, dtype=float)
    if density.ndim != 1:
        raise ValueError("ys must be one-dimensional")
    if areas.ndim == 0:
        areas = np.full(density.size, float(areas), dtype=float)
    if areas.shape != density.shape:
        raise ValueError("element_areas must match ys")
    if not np.all(np.isfinite(density)) or not np.all(np.isfinite(areas)):
        raise ValueError("ys and element_areas must be finite")
    if np.any(density < -1e-12) or np.any(density > 1.0 + 1e-12):
        raise ValueError("ys values must be in [0, 1]")
    if np.any(areas <= 0.0):
        raise ValueError("element_areas must be positive")
    clipped = np.clip(density, 0.0, 1.0)
    product = clipped * (1.0 - clipped)
    total_area = float(np.sum(areas))
    global_grayness = float(4.0 * np.sum(areas * product) / total_area)
    max_element_grayness = float(np.max(np.minimum(clipped, 1.0 - clipped)))
    gray_count = int(np.count_nonzero((clipped > 1e-3) & (clipped < 1.0 - 1e-3)))
    return GDPHullGraynessMetrics(
        global_grayness=global_grayness,
        max_element_grayness=max_element_grayness,
        gray_element_count=gray_count,
        mean_solid_fraction=float(np.mean(clipped)),
    )


def round_relaxed_density_to_binary(ys, volfrac: float) -> np.ndarray:
    density = np.asarray(ys, dtype=float)
    if density.ndim != 1:
        raise ValueError("ys must be one-dimensional")
    if not np.all(np.isfinite(density)):
        raise ValueError("ys must contain only finite values")
    if not np.isfinite(volfrac) or not (0.0 < volfrac <= 1.0):
        raise ValueError("volfrac must satisfy 0 < volfrac <= 1")
    solid_count = int(np.floor(float(volfrac) * density.size + 1e-12))
    binary = np.zeros(density.size, dtype=float)
    if solid_count > 0:
        order = np.argsort(-density, kind="mergesort")
        binary[order[:solid_count]] = 1.0
    return binary


def _binary_config(config: GDPHullRelaxationConfig) -> GDPHullBinaryConfig:
    return GDPHullBinaryConfig(
        benchmark=config.benchmark,
        nelx=config.nelx,
        nely=config.nely,
        length=config.length,
        height=config.height,
        volfrac=config.volfrac,
        displacement_bound=config.displacement_bound,
        e0=config.e0,
        nu=config.nu,
    )


def _build_data(config: GDPHullRelaxationConfig) -> GDPHullRelaxationModelData:
    mesh = build_benchmark_mesh(_binary_config(config))
    element_ke = q4_plane_stress_stiffness(
        width=config.length / config.nelx,
        height=config.height / config.nely,
        young=config.e0,
        nu=config.nu,
    )
    free_dof_to_index = _free_dof_to_index(mesh)
    equilibrium_terms = _build_solid_phase_equilibrium_terms(mesh, element_ke, free_dof_to_index)
    try:
        _, full_solid_displacement = solve_linear_response(
            mesh,
            element_ke,
            np.ones(mesh.element_count, dtype=float),
        )
        full_solid_compliance = float(mesh.force @ full_solid_displacement)
        if not np.isfinite(full_solid_compliance) or full_solid_compliance <= 0.0:
            raise RuntimeError("full-solid compliance is not positive and finite")
    except RuntimeError:
        full_solid_displacement = np.zeros(mesh.total_dofs, dtype=float)
        full_solid_compliance = 0.0
    if config.use_tight_displacement_bounds:
        absolute_displacement = np.abs(full_solid_displacement)
        max_full_displacement = float(np.max(absolute_displacement))
        floor = max(
            max_full_displacement * float(config.tight_displacement_bound_floor_fraction),
            1e-9,
        )
        displacement_bounds = (
            float(config.tight_displacement_bound_safety_factor)
            * np.maximum(absolute_displacement, floor)
        )
        displacement_bounds = np.minimum(displacement_bounds, float(config.displacement_bound))
    else:
        displacement_bounds = np.full(mesh.total_dofs, float(config.displacement_bound), dtype=float)
    reference_density = np.full(mesh.element_count, max(float(config.volfrac), 1e-3), dtype=float)
    try:
        _, reference_displacement = solve_linear_response(mesh, element_ke, reference_density)
        reference_compliance = max(float(mesh.force @ reference_displacement), 1e-8)
    except RuntimeError:
        reference_compliance = 1.0
    phase_filter = (
        None
        if config.phase_filter_radius is None
        else build_density_filter(mesh, float(config.phase_filter_radius))
    )
    return GDPHullRelaxationModelData(
        mesh=mesh,
        element_ke=element_ke,
        config=config,
        free_dof_to_index=free_dof_to_index,
        equilibrium_terms=equilibrium_terms,
        reference_compliance=reference_compliance,
        element_directed_edges=_element_directed_edges(mesh),
        support_elements=_support_elements(mesh),
        load_elements=_load_elements(mesh),
        full_solid_compliance=full_solid_compliance,
        full_solid_displacement=full_solid_displacement,
        displacement_bounds=displacement_bounds,
        phase_filter=phase_filter,
    )


def _add_element_connectivity_constraints(
    model: pyo.ConcreteModel,
    data: GDPHullRelaxationModelData,
) -> None:
    mesh = data.mesh
    if not data.support_elements:
        raise ValueError("element connectivity requires at least one support-adjacent element")
    if not data.load_elements:
        raise ValueError("element connectivity requires at least one load-adjacent element")

    edge_in: list[list[tuple[int, int]]] = [[] for _ in range(mesh.element_count)]
    edge_out: list[list[tuple[int, int]]] = [[] for _ in range(mesh.element_count)]
    for source, target in data.element_directed_edges:
        edge_out[source].append((source, target))
        edge_in[target].append((source, target))

    model.CA = pyo.Set(initialize=data.element_directed_edges, dimen=2)
    model.CR = pyo.Set(initialize=data.support_elements)
    model.CL = pyo.Set(initialize=data.load_elements)
    model.connectivity_flow = pyo.Var(model.CA, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
    model.connectivity_source_flow = pyo.Var(
        model.CR,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, 1.0),
    )
    model.connectivity_load_sink = pyo.Var(
        model.CL,
        domain=pyo.NonNegativeReals,
        bounds=(0.0, 1.0),
    )

    def arc_tail_capacity_rule(m: pyo.ConcreteModel, source: int, target: int):
        return m.connectivity_flow[source, target] <= m.solid_phase[source]

    def arc_head_capacity_rule(m: pyo.ConcreteModel, source: int, target: int):
        return m.connectivity_flow[source, target] <= m.solid_phase[target]

    def source_capacity_rule(m: pyo.ConcreteModel, element: int):
        return m.connectivity_source_flow[element] <= m.solid_phase[element]

    def sink_capacity_rule(m: pyo.ConcreteModel, element: int):
        return m.connectivity_load_sink[element] <= m.solid_phase[element]

    def flow_balance_rule(m: pyo.ConcreteModel, element: int):
        source_flow = (
            m.connectivity_source_flow[element]
            if element in data.support_elements
            else 0.0
        )
        sink_flow = (
            m.connectivity_load_sink[element]
            if element in data.load_elements
            else 0.0
        )
        incoming = sum(m.connectivity_flow[source, target] for source, target in edge_in[int(element)])
        outgoing = sum(m.connectivity_flow[source, target] for source, target in edge_out[int(element)])
        return source_flow + incoming - outgoing >= sink_flow

    model.connectivity_arc_tail_capacity = pyo.Constraint(model.CA, rule=arc_tail_capacity_rule)
    model.connectivity_arc_head_capacity = pyo.Constraint(model.CA, rule=arc_head_capacity_rule)
    model.connectivity_source_capacity = pyo.Constraint(model.CR, rule=source_capacity_rule)
    model.connectivity_load_sink_capacity = pyo.Constraint(model.CL, rule=sink_capacity_rule)
    model.connectivity_flow_balance = pyo.Constraint(model.E, rule=flow_balance_rule)
    model.connectivity_load_sink_demand = pyo.Constraint(
        expr=sum(model.connectivity_load_sink[e] for e in data.load_elements) >= 1.0
    )
    model.load_connected = pyo.Constraint(
        expr=sum(model.solid_phase[e] for e in data.load_elements) >= 1.0
    )


def _add_gdp_hull_phase_displacement_map_expressions(
    model: pyo.ConcreteModel,
    data: GDPHullRelaxationModelData,
    stage: GDPHullRelaxationStage,
) -> None:
    mesh = data.mesh

    def solid_phase_scale_rule(m: pyo.ConcreteModel, e: int):
        y = m.solid_phase[e]
        if stage.solid_displacement_coupling_phase_law == "simp":
            return y ** float(stage.solid_displacement_coupling_phase_exponent)
        if stage.solid_displacement_coupling_phase_law == "ramp":
            q = float(stage.solid_displacement_coupling_ramp_q)
            return y / (1.0 + q * (1.0 - y))
        return y

    # GDP phase contribution map: the solid disjunct contributes scaled element
    # displacement in the continuous Hull relaxation.
    model.gdp_solid_phase_scale = pyo.Expression(model.E, rule=solid_phase_scale_rule)
    model.__dict__["solid_phase_stiffness"] = model.gdp_solid_phase_scale
    model.gdp_solid_displacement_map_residual = pyo.Expression(
        model.E,
        model.A,
        rule=lambda m, e, a: m.us[e, a]
        - m.gdp_solid_phase_scale[e]
        * _global_displacement_expression(m, data, int(mesh.element_dofs[int(e), int(a)])),
    )
    model.__dict__["solid_displacement_coupling_residual"] = (
        model.gdp_solid_displacement_map_residual
    )


def _add_gdp_hull_phase_displacement_map_constraints(
    model: pyo.ConcreteModel,
    stage: GDPHullRelaxationStage,
) -> None:
    if stage.solid_displacement_coupling_mode == "equality":
        model.gdp_solid_displacement_map_equalities = pyo.Constraint(
            model.E,
            model.A,
            rule=lambda m, e, a: m.gdp_solid_displacement_map_residual[e, a] == 0.0,
        )
        model.__dict__["solid_displacement_coupling_equalities"] = (
            model.gdp_solid_displacement_map_equalities
        )


def build_gdp_hull_relaxation_model(
    config: GDPHullRelaxationConfig,
    stage: GDPHullRelaxationStage,
    initial: GDPHullRelaxationSolution | None = None,
) -> tuple[pyo.ConcreteModel, GDPHullRelaxationModelData]:
    _validate_config(config)
    _validate_stage(stage)
    data = _build_data(config)
    mesh = data.mesh
    displacement_bounds = np.asarray(data.displacement_bounds, dtype=float)
    phase_projection_enabled = data.phase_filter is not None or stage.heaviside_beta is not None
    if phase_projection_enabled and data.phase_filter is None:
        raise ValueError("phase_filter_radius is required when heaviside_beta is set")
    if phase_projection_enabled and stage.heaviside_beta is None:
        raise ValueError("heaviside_beta is required when phase_filter_radius is set")

    model = pyo.ConcreteModel()
    model.E = pyo.RangeSet(0, mesh.element_count - 1)
    model.A = pyo.RangeSet(0, 7)
    model.F = pyo.RangeSet(0, mesh.free_dofs.size - 1)

    init_ys = np.full(mesh.element_count, float(config.volfrac), dtype=float)
    init_u = np.zeros(mesh.free_dofs.size, dtype=float)
    init_us = np.zeros((mesh.element_count, 8), dtype=float)
    if initial is not None:
        initial_raw_ys = getattr(initial, "raw_ys", None)
        if initial_raw_ys is None:
            initial_raw_ys = initial.ys
        init_ys = np.clip(np.asarray(initial_raw_ys, dtype=float), 1e-6, 1.0 - 1e-6)
        init_u = np.asarray(initial.free_displacement, dtype=float)
    init_u = np.clip(init_u, -displacement_bounds[mesh.free_dofs], displacement_bounds[mesh.free_dofs])

    def global_displacement_bound(global_dof: int) -> float:
        return float(displacement_bounds[int(global_dof)])

    def free_displacement_bound(free_index: int) -> float:
        return global_displacement_bound(int(mesh.free_dofs[int(free_index)]))

    def element_displacement_bound(element_index: int, local_index: int) -> float:
        return global_displacement_bound(int(mesh.element_dofs[int(element_index), int(local_index)]))

    if initial is not None:
        initial_us = getattr(initial, "solid_displacement_components", None)
        if initial_us is not None:
            init_us_array = np.asarray(initial_us, dtype=float)
            if init_us_array.shape == init_us.shape and np.all(np.isfinite(init_us_array)):
                for element_index in range(mesh.element_count):
                    for local_index in range(8):
                        bound = element_displacement_bound(element_index, local_index)
                        init_us[element_index, local_index] = float(
                            np.clip(init_us_array[element_index, local_index], -bound, bound)
                        )

    model.ys = pyo.Var(model.E, domain=pyo.UnitInterval, initialize=lambda _, e: float(init_ys[int(e)]))
    model.yv = pyo.Var(model.E, domain=pyo.UnitInterval, initialize=lambda _, e: float(1.0 - init_ys[int(e)]))
    model.u = pyo.Var(
        model.F,
        bounds=lambda _, i: (-free_displacement_bound(i), free_displacement_bound(i)),
        initialize=lambda _, i: float(init_u[int(i)]),
    )
    model.us = pyo.Var(
        model.E,
        model.A,
        bounds=lambda _, e, a: (
            -element_displacement_bound(e, a),
            element_displacement_bound(e, a),
        ),
        initialize=lambda _, e, a: float(init_us[int(e), int(a)]),
    )
    if config.fix_load_elements:
        required_fraction = len(data.load_elements) / float(mesh.element_count)
        if required_fraction > float(config.volfrac) + 1e-12:
            raise ValueError("fix_load_elements requires more load-adjacent elements than the volume fraction allows")
        for element_index in data.load_elements:
            model.ys[element_index].fix(1.0)
            model.yv[element_index].fix(0.0)

    model.phase_sum = pyo.Constraint(model.E, rule=lambda m, e: m.ys[e] + m.yv[e] == 1.0)
    if phase_projection_enabled:
        assert data.phase_filter is not None
        beta = float(stage.heaviside_beta)
        eta = float(config.heaviside_eta)
        denominator = float(np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta)))
        lower_numerator = float(np.tanh(beta * eta))
        init_tilde = np.zeros(mesh.element_count, dtype=float)
        for row_index, row in enumerate(data.phase_filter.rows):
            init_tilde[row_index] = sum(weight * init_ys[col] for col, weight in row)
        init_bar = heaviside_projection(init_tilde, beta=beta, eta=eta)

        def filtered_solid_rule(m: pyo.ConcreteModel, e: int):
            return sum(weight * m.ys[col] for col, weight in data.phase_filter.rows[int(e)])

        model.ys_tilde = pyo.Var(
            model.E,
            domain=pyo.UnitInterval,
            initialize=lambda _, e: float(init_tilde[int(e)]),
        )
        model.ys_bar = pyo.Var(
            model.E,
            domain=pyo.UnitInterval,
            initialize=lambda _, e: float(init_bar[int(e)]),
        )
        model.phase_filter_definition = pyo.Constraint(
            model.E,
            rule=lambda m, e: m.ys_tilde[e] == filtered_solid_rule(m, e),
        )
        model.heaviside_projection_definition = pyo.Constraint(
            model.E,
            rule=lambda m, e: m.ys_bar[e]
            == (lower_numerator + pyo.tanh(beta * (m.ys_tilde[e] - eta))) / denominator,
        )
        model.yv_tilde = pyo.Expression(model.E, rule=lambda m, e: 1.0 - m.ys_tilde[e])
        model.yv_bar = pyo.Expression(model.E, rule=lambda m, e: 1.0 - m.ys_bar[e])
        model.solid_phase = pyo.Expression(model.E, rule=lambda m, e: m.ys_bar[e])
        model.void_phase = pyo.Expression(model.E, rule=lambda m, e: m.yv_bar[e])
    else:
        model.solid_phase = pyo.Expression(model.E, rule=lambda m, e: m.ys[e])
        model.void_phase = pyo.Expression(model.E, rule=lambda m, e: m.yv[e])

    model.solid_lower = pyo.Constraint(
        model.E,
        model.A,
        rule=lambda m, e, a: -element_displacement_bound(e, a) * m.solid_phase[e] <= m.us[e, a],
    )
    model.solid_upper = pyo.Constraint(
        model.E,
        model.A,
        rule=lambda m, e, a: m.us[e, a] <= element_displacement_bound(e, a) * m.solid_phase[e],
    )

    def void_lower_rule(m: pyo.ConcreteModel, e: int, a: int):
        global_dof = int(mesh.element_dofs[int(e), int(a)])
        return (
            -global_displacement_bound(global_dof) * m.void_phase[e]
            <= _global_displacement_expression(m, data, global_dof) - m.us[e, a]
        )

    def void_upper_rule(m: pyo.ConcreteModel, e: int, a: int):
        global_dof = int(mesh.element_dofs[int(e), int(a)])
        return (
            _global_displacement_expression(m, data, global_dof) - m.us[e, a]
            <= global_displacement_bound(global_dof) * m.void_phase[e]
        )

    model.void_lower = pyo.Constraint(model.E, model.A, rule=void_lower_rule)
    model.void_upper = pyo.Constraint(model.E, model.A, rule=void_upper_rule)
    _add_gdp_hull_phase_displacement_map_expressions(model, data, stage)

    def equilibrium_rule(m: pyo.ConcreteModel, row: int):
        terms = data.equilibrium_terms[int(row)]
        force = float(mesh.force[int(mesh.free_dofs[int(row)])])
        return sum(value * m.us[element_index, local_col] for element_index, local_col, value in terms) == force

    model.equilibrium = pyo.Constraint(model.F, rule=equilibrium_rule)
    model.volume_constraint = pyo.Constraint(
        expr=sum(model.solid_phase[e] * mesh.element_area for e in model.E)
        == config.volfrac * mesh.element_count * mesh.element_area
    )
    model.compliance = pyo.Expression(
        expr=sum(float(mesh.force[int(global_dof)]) * model.u[i] for i, global_dof in enumerate(mesh.free_dofs))
    )
    if config.enforce_full_solid_compliance_lower_bound:
        if data.full_solid_compliance <= 0.0:
            raise RuntimeError("full-solid compliance lower bound requires a valid full-solid response")
        model.full_solid_compliance_lower_bound = pyo.Constraint(
            expr=model.compliance >= float(data.full_solid_compliance)
        )
    loaded_free_dofs = [
        (index, float(mesh.force[int(global_dof)]))
        for index, global_dof in enumerate(mesh.free_dofs)
        if float(mesh.force[int(global_dof)]) != 0.0
    ]
    model.L = pyo.RangeSet(0, len(loaded_free_dofs) - 1)

    def load_direction_rule(m: pyo.ConcreteModel, load_index: int):
        free_index, force_value = loaded_free_dofs[int(load_index)]
        return force_value * m.u[free_index] >= 0.0

    model.load_direction = pyo.Constraint(model.L, rule=load_direction_rule)
    if config.use_element_connectivity:
        _add_element_connectivity_constraints(model, data)
    model.grayness = pyo.Expression(
        expr=(4.0 / mesh.element_count) * sum(model.solid_phase[e] * model.void_phase[e] for e in model.E)
    )
    if stage.grayness_limit is not None:
        model.grayness_constraint = pyo.Constraint(expr=model.grayness <= float(stage.grayness_limit))
    _add_gdp_hull_phase_displacement_map_constraints(model, stage)
    if stage.solid_displacement_coupling_norm == "energy":
        model.solid_displacement_coupling = pyo.Expression(
            expr=sum(
                sum(
                    model.solid_displacement_coupling_residual[e, a]
                    * float(data.element_ke[int(a), int(b)])
                    * model.solid_displacement_coupling_residual[e, b]
                    for a in model.A
                    for b in model.A
                )
                for e in model.E
            )
            / max(float(data.reference_compliance), 1e-12)
        )
    else:
        model.solid_displacement_coupling = pyo.Expression(
            expr=(1.0 / (mesh.element_count * 8.0))
            * sum(
                (
                    model.solid_displacement_coupling_residual[e, a]
                    / max(element_displacement_bound(e, a), 1e-12)
                )
                ** 2
                for e in model.E
                for a in model.A
            )
        )
    model.solid_displacement_coupling_penalty = float(stage.solid_displacement_coupling_penalty)
    model.objective = pyo.Objective(
        expr=(
            model.compliance / data.reference_compliance
            + float(stage.penalty) * model.grayness
            + float(stage.solid_displacement_coupling_penalty) * model.solid_displacement_coupling
        ),
        sense=pyo.minimize,
    )
    return model, data


def recompute_rounded_binary_response(
    data: GDPHullRelaxationModelData,
    binary_density: np.ndarray,
) -> tuple[float | None, np.ndarray | None, str]:
    density = np.asarray(binary_density, dtype=float)
    try:
        displacement = _solve_active_binary_linear_response(data, density)
    except RuntimeError as exc:
        return None, None, f"failed: {exc}"
    compliance = float(data.mesh.force @ displacement)
    if not np.isfinite(compliance) or compliance <= 0.0:
        return None, displacement, "failed: nonphysical compliance"
    max_displacement = float(np.max(np.abs(displacement)))
    saturation = max_displacement / float(data.config.displacement_bound)
    if not np.isfinite(max_displacement) or not np.isfinite(saturation):
        return None, displacement, "failed: non-finite displacement"
    if saturation >= 0.95:
        return None, displacement, f"failed: displacement bound saturation {saturation:.6g} >= 0.95"
    return compliance, displacement, "ok"


def _read_var_vector(indexed_var: pyo.Var, indices, expected_size: int, name: str) -> np.ndarray:
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


def _read_var_matrix(indexed_var: pyo.Var, row_indices, col_indices, shape: tuple[int, int], name: str) -> np.ndarray:
    values = np.zeros(shape, dtype=float)
    for row in row_indices:
        for col in col_indices:
            raw = pyo.value(indexed_var[row, col], exception=False)
            if raw is None:
                raise RuntimeError(f"{name}[{row}, {col}] has no value")
            value = float(raw)
            if not np.isfinite(value):
                raise RuntimeError(f"{name}[{row}, {col}] must be finite")
            values[int(row), int(col)] = value
    return values


def _validate_solution_vector(values, expected_size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def create_gdp_hull_relaxation_initial_solution(
    config: GDPHullRelaxationConfig,
    ys,
    full_displacement=None,
    solid_displacement_components=None,
    raw_ys=None,
    raw_yv=None,
    filtered_ys=None,
) -> GDPHullRelaxationSolution:
    _validate_config(config)
    data = _build_data(config)
    mesh = data.mesh
    density = np.clip(_validate_solution_vector(ys, mesh.element_count, "ys"), 0.0, 1.0)
    void_density = 1.0 - density
    raw_solid = density.copy() if raw_ys is None else np.clip(
        _validate_solution_vector(raw_ys, mesh.element_count, "raw_ys"),
        0.0,
        1.0,
    )
    raw_void = 1.0 - raw_solid if raw_yv is None else np.clip(
        _validate_solution_vector(raw_yv, mesh.element_count, "raw_yv"),
        0.0,
        1.0,
    )
    filtered_solid = None
    if filtered_ys is not None:
        filtered_solid = np.clip(
            _validate_solution_vector(filtered_ys, mesh.element_count, "filtered_ys"),
            0.0,
            1.0,
        )
    if full_displacement is None:
        displacement = np.zeros(mesh.total_dofs, dtype=float)
    else:
        displacement = _validate_solution_vector(full_displacement, mesh.total_dofs, "full_displacement")
    if solid_displacement_components is None:
        solid_components = np.zeros((mesh.element_count, 8), dtype=float)
    else:
        solid_components = np.asarray(solid_displacement_components, dtype=float)
        if solid_components.shape != (mesh.element_count, 8):
            raise ValueError(f"solid_displacement_components must have shape ({mesh.element_count}, 8)")
        if not np.all(np.isfinite(solid_components)):
            raise ValueError("solid_displacement_components must contain only finite values")
    compliance = float(mesh.force @ displacement)
    metrics = grayness_metrics(density, np.full(mesh.element_count, mesh.element_area, dtype=float))
    binary = round_relaxed_density_to_binary(density, config.volfrac)
    rounded_compliance, rounded_displacement, rounded_status = recompute_rounded_binary_response(data, binary)
    return GDPHullRelaxationSolution(
        ys=density,
        yv=void_density,
        free_displacement=displacement[mesh.free_dofs],
        full_displacement=displacement,
        solid_displacement_components=solid_components,
        continuous_compliance=compliance,
        volume_fraction=float(np.mean(density)),
        global_grayness=metrics.global_grayness,
        max_element_grayness=metrics.max_element_grayness,
        stage_results=[],
        rounded_binary_density=binary,
        rounded_binary_compliance=rounded_compliance,
        rounded_binary_volume_fraction=float(np.mean(binary)),
        rounded_binary_displacement=rounded_displacement,
        rounded_response_status=rounded_status,
        model_data=data,
        raw_ys=raw_solid,
        raw_yv=raw_void,
        filtered_ys=filtered_solid,
    )


def _extract_stage_solution(
    model: pyo.ConcreteModel,
    data: GDPHullRelaxationModelData,
    results: Any,
    stage: GDPHullRelaxationStage,
    stage_index: int,
    elapsed_seconds: float,
    prior_stage_results: list[GDPHullRelaxationStageResult],
) -> GDPHullRelaxationSolution:
    mesh = data.mesh
    raw_ys = np.clip(_read_var_vector(model.ys, model.E, mesh.element_count, "ys"), 0.0, 1.0)
    raw_yv = np.clip(_read_var_vector(model.yv, model.E, mesh.element_count, "yv"), 0.0, 1.0)
    ys = np.clip(_read_var_vector(model.solid_phase, model.E, mesh.element_count, "solid_phase"), 0.0, 1.0)
    yv = np.clip(_read_var_vector(model.void_phase, model.E, mesh.element_count, "void_phase"), 0.0, 1.0)
    if hasattr(model, "ys_tilde"):
        filtered_ys = np.clip(_read_var_vector(model.ys_tilde, model.E, mesh.element_count, "ys_tilde"), 0.0, 1.0)
    else:
        filtered_ys = raw_ys.copy()
    free_displacement = _read_var_vector(model.u, model.F, mesh.free_dofs.size, "u")
    full_displacement = np.zeros(mesh.total_dofs, dtype=float)
    full_displacement[mesh.free_dofs] = free_displacement
    solid_displacement_components = _read_var_matrix(
        model.us,
        model.E,
        model.A,
        (mesh.element_count, 8),
        "us",
    )
    compliance = float(mesh.force @ full_displacement)
    metrics = grayness_metrics(ys, np.full(mesh.element_count, mesh.element_area, dtype=float))
    coupling = float(pyo.value(model.solid_displacement_coupling, exception=False) or 0.0)
    normalized_objective = float(
        compliance / data.reference_compliance
        + float(stage.penalty) * metrics.global_grayness
        + float(stage.solid_displacement_coupling_penalty) * coupling
    )
    stage_result = GDPHullRelaxationStageResult(
        stage_index=stage_index,
        penalty=float(stage.penalty),
        grayness_limit=None if stage.grayness_limit is None else float(stage.grayness_limit),
        heaviside_beta=None if stage.heaviside_beta is None else float(stage.heaviside_beta),
        compliance=compliance,
        normalized_objective=normalized_objective,
        global_grayness=metrics.global_grayness,
        max_element_grayness=metrics.max_element_grayness,
        volume_fraction=float(np.mean(ys)),
        solid_displacement_coupling_penalty=float(stage.solid_displacement_coupling_penalty),
        solid_displacement_coupling=coupling,
        solid_displacement_coupling_phase_law=str(stage.solid_displacement_coupling_phase_law),
        solid_displacement_coupling_phase_exponent=float(stage.solid_displacement_coupling_phase_exponent),
        solid_displacement_coupling_ramp_q=float(stage.solid_displacement_coupling_ramp_q),
        solid_displacement_coupling_norm=str(stage.solid_displacement_coupling_norm),
        solid_displacement_coupling_mode=str(stage.solid_displacement_coupling_mode),
        solver_status=str(results.solver.status),
        termination_condition=str(results.solver.termination_condition),
        elapsed_seconds=float(elapsed_seconds),
    )
    binary = round_relaxed_density_to_binary(ys, data.config.volfrac)
    rounded_compliance, rounded_displacement, rounded_status = recompute_rounded_binary_response(data, binary)
    return GDPHullRelaxationSolution(
        ys=ys,
        yv=yv,
        free_displacement=free_displacement,
        full_displacement=full_displacement,
        solid_displacement_components=solid_displacement_components,
        continuous_compliance=compliance,
        volume_fraction=float(np.mean(ys)),
        global_grayness=metrics.global_grayness,
        max_element_grayness=metrics.max_element_grayness,
        stage_results=[*prior_stage_results, stage_result],
        rounded_binary_density=binary,
        rounded_binary_compliance=rounded_compliance,
        rounded_binary_volume_fraction=float(np.mean(binary)),
        rounded_binary_displacement=rounded_displacement,
        rounded_response_status=rounded_status,
        model_data=data,
        raw_ys=raw_ys,
        raw_yv=raw_yv,
        filtered_ys=filtered_ys,
    )


def solve_gdp_hull_relaxation_stage(
    config: GDPHullRelaxationConfig,
    stage: GDPHullRelaxationStage,
    initial: GDPHullRelaxationSolution | None = None,
    stage_index: int = 0,
    prior_stage_results: list[GDPHullRelaxationStageResult] | None = None,
) -> GDPHullRelaxationSolution:
    model, data = build_gdp_hull_relaxation_model(config, stage, initial=initial)
    solver = create_ipopt_solver(
        SolverSettings(
            tol=config.ipopt_tol,
            max_iter=int(config.ipopt_max_iter),
            tee=config.tee,
        )
    )
    if not solver.available(False):
        raise RuntimeError("IPOPT solver is not available")
    started_at = time.perf_counter()
    results = solver.solve(model, tee=config.tee)
    elapsed = time.perf_counter() - started_at
    termination = results.solver.termination_condition
    acceptable = {
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.feasible,
        TerminationCondition.maxIterations,
        TerminationCondition.maxTimeLimit,
    }
    if termination not in acceptable:
        raise RuntimeError(
            f"GDP-Hull relaxation solve failed: status={results.solver.status}, termination={termination}"
        )
    return _extract_stage_solution(
        model,
        data,
        results,
        stage,
        stage_index,
        elapsed,
        list(prior_stage_results or []),
    )


def solve_gdp_hull_relaxation_continuation(
    config: GDPHullRelaxationConfig,
    stages: list[GDPHullRelaxationStage] | tuple[GDPHullRelaxationStage, ...],
    initial: GDPHullRelaxationSolution | None = None,
) -> GDPHullRelaxationSolution:
    if not stages:
        raise ValueError("stages must contain at least one stage")
    solution = initial
    history: list[GDPHullRelaxationStageResult] = []
    for stage_index, stage in enumerate(stages):
        try:
            solution = solve_gdp_hull_relaxation_stage(
                config,
                stage,
                initial=solution,
                stage_index=stage_index,
                prior_stage_results=history,
            )
        except RuntimeError as exc:
            if solution is None:
                raise
            solution.completed_all_stages = False
            solution.failed_stage_index = stage_index
            solution.failed_stage_reason = str(exc)
            return solution
        history = list(solution.stage_results)
    if solution is None:
        raise RuntimeError("relaxation continuation did not produce a solution")
    solution.completed_all_stages = True
    solution.failed_stage_index = None
    solution.failed_stage_reason = None
    return solution


__all__ = [
    "GDPHullGraynessMetrics",
    "GDPHullRelaxationConfig",
    "GDPHullRelaxationModelData",
    "GDPHullRelaxationSolution",
    "GDPHullRelaxationStage",
    "GDPHullRelaxationStageResult",
    "build_gdp_hull_relaxation_model",
    "create_gdp_hull_relaxation_initial_solution",
    "grayness_metrics",
    "heaviside_projection",
    "recompute_rounded_binary_response",
    "round_relaxed_density_to_binary",
    "solve_gdp_hull_relaxation_continuation",
    "solve_gdp_hull_relaxation_stage",
]
