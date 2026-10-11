from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination

from tpo.element import q4_plane_stress_stiffness
from tpo.fea import build_equilibrium_terms, solve_linear_response
from tpo.filtering import DensityFilter, apply_density_filter, build_density_filter
from tpo.mesh import Mesh, build_cantilever_mesh, build_lbracket_mesh, build_mbb_mesh
from tpo.runtime import SolverSettings, create_ipopt_solver


@dataclass(frozen=True)
class SimpConfig:
    benchmark: str = "cantilever"
    nelx: int = 30
    nely: int = 10
    length: float = 3.0
    height: float = 1.0
    volfrac: float = 0.4
    rmin: float = 1.5
    xmin: float = 1e-3
    e0: float = 1.0
    emin: float = 1e-6
    nu: float = 0.3
    penalties: tuple[float, ...] = (1.0, 2.0, 3.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", _canonical_benchmark_label(self.benchmark))
        try:
            penalties = tuple(self.penalties)
        except TypeError:
            return
        object.__setattr__(self, "penalties", penalties)


@dataclass(frozen=True)
class SimpModelData:
    mesh: Mesh
    element_ke: np.ndarray
    filter_data: DensityFilter
    row_terms: list[list[tuple[int, int, float]]]
    config: SimpConfig
    penalty: float


@dataclass
class SimpStageResult:
    penalty: float
    compliance: float
    volume_fraction: float
    max_displacement: float
    solver_status: str
    termination_condition: str
    elapsed_seconds: float


@dataclass
class SimpStageSnapshot:
    penalty: float
    raw_density: np.ndarray
    physical_density: np.ndarray
    free_displacement: np.ndarray
    full_displacement: np.ndarray


@dataclass
class SimpSolution:
    raw_density: np.ndarray
    physical_density: np.ndarray
    free_displacement: np.ndarray
    full_displacement: np.ndarray
    stage_results: list[SimpStageResult]
    model_data: SimpModelData
    stage_snapshots: list[SimpStageSnapshot] = field(default_factory=list)


def _validate_penalties(penalties) -> tuple[float, ...]:
    try:
        penalty_values = tuple(float(value) for value in penalties)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "penalties must be a non-empty iterable of finite positive values"
        ) from exc
    if len(penalty_values) == 0:
        raise ValueError("penalties must contain at least one value")
    if not all(np.isfinite(value) and value > 0.0 for value in penalty_values):
        raise ValueError("penalties must contain only finite positive values")
    return penalty_values


def _canonical_benchmark_label(benchmark: str) -> str:
    label = str(benchmark).strip().lower().replace("-", "_")
    aliases = {
        "cantilever": "cantilever",
        "mbb": "mbb",
        "lbracket": "lbracket",
        "l_bracket": "lbracket",
        "lbeam": "lbracket",
        "l_beam": "lbracket",
    }
    return aliases.get(label, label)


def build_benchmark_mesh(config: SimpConfig) -> Mesh:
    benchmark = _canonical_benchmark_label(config.benchmark)
    if benchmark == "cantilever":
        return build_cantilever_mesh(
            nelx=config.nelx,
            nely=config.nely,
            length=config.length,
            height=config.height,
        )
    if benchmark == "mbb":
        return build_mbb_mesh(
            nelx=config.nelx,
            nely=config.nely,
            length=config.length,
            height=config.height,
        )
    if benchmark == "lbracket":
        return build_lbracket_mesh(
            nelx=config.nelx,
            nely=config.nely,
            length=config.length,
            height=config.height,
        )
    raise ValueError(
        "unknown benchmark; expected one of cantilever, mbb, or lbracket"
    )


def _validate_config(config: SimpConfig, penalty: float) -> None:
    if not np.isfinite(config.volfrac) or not (0.0 < config.volfrac <= 1.0):
        raise ValueError("volfrac must be finite and satisfy 0.0 < volfrac <= 1.0")
    if not np.isfinite(config.xmin) or not (0.0 < config.xmin <= 1.0):
        raise ValueError("xmin must be finite and satisfy 0.0 < xmin <= 1.0")
    if config.xmin > config.volfrac:
        raise ValueError("xmin must be less than or equal to volfrac")
    if not np.isfinite(config.e0) or config.e0 <= 0.0:
        raise ValueError("e0 must be finite and greater than 0.0")
    if not np.isfinite(config.emin) or not (0.0 <= config.emin < config.e0):
        raise ValueError("emin must be finite and satisfy 0.0 <= emin < e0")
    if not np.isfinite(penalty) or penalty <= 0.0:
        raise ValueError("penalty must be finite and greater than 0.0")
    _validate_penalties(config.penalties)


def _validate_initial_density(
    initial_density: np.ndarray,
    mesh: Mesh,
    config: SimpConfig,
) -> np.ndarray:
    try:
        density = np.asarray(initial_density, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_density must be a finite NumPy-compatible array") from exc
    if density.ndim != 1:
        raise ValueError("initial_density must be one-dimensional")
    if density.size != mesh.element_count:
        raise ValueError(f"initial_density length must be {mesh.element_count}")
    if not np.all(np.isfinite(density)):
        raise ValueError("initial_density must contain only finite values")
    if np.any((density < config.xmin) | (density > 1.0)):
        raise ValueError("initial_density values must be within [xmin, 1.0]")
    return density


def _validate_initial_free_displacement(
    initial_free_displacement: np.ndarray,
    mesh: Mesh,
) -> np.ndarray:
    try:
        displacement = np.asarray(initial_free_displacement, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "initial_free_displacement must be a finite NumPy-compatible array"
        ) from exc
    if displacement.ndim != 1:
        raise ValueError("initial_free_displacement must be one-dimensional")
    if displacement.size != mesh.free_dofs.size:
        raise ValueError(
            f"initial_free_displacement length must be {mesh.free_dofs.size}"
        )
    if not np.all(np.isfinite(displacement)):
        raise ValueError("initial_free_displacement must contain only finite values")
    return displacement


def _initial_response(
    mesh: Mesh,
    element_ke: np.ndarray,
    config: SimpConfig,
    raw_density: np.ndarray,
    filter_data: DensityFilter,
    penalty: float,
) -> np.ndarray:
    physical = apply_density_filter(filter_data, raw_density)
    stiffness = config.emin + (config.e0 - config.emin) * physical**penalty
    free_u, _ = solve_linear_response(mesh, element_ke, stiffness)
    return free_u


def build_simp_model(
    config: SimpConfig,
    penalty: float,
    initial_density: np.ndarray | None = None,
    initial_free_displacement: np.ndarray | None = None,
) -> tuple[pyo.ConcreteModel, SimpModelData]:
    _validate_config(config, penalty)

    mesh = build_benchmark_mesh(config)
    element_ke = q4_plane_stress_stiffness(
        width=config.length / config.nelx,
        height=config.height / config.nely,
        young=1.0,
        nu=config.nu,
    )
    filter_data = build_density_filter(mesh, radius_elements=config.rmin)
    row_terms = build_equilibrium_terms(mesh, element_ke)

    if initial_density is None:
        initial_density = np.full(mesh.element_count, config.volfrac, dtype=float)
    else:
        initial_density = _validate_initial_density(initial_density, mesh, config)
    if initial_free_displacement is None:
        initial_free_displacement = _initial_response(
            mesh, element_ke, config, initial_density, filter_data, penalty
        )
    else:
        initial_free_displacement = _validate_initial_free_displacement(
            initial_free_displacement, mesh
        )

    model = pyo.ConcreteModel()
    model.E = pyo.RangeSet(0, mesh.element_count - 1)
    model.F = pyo.RangeSet(0, mesh.free_dofs.size - 1)
    model.x = pyo.Var(
        model.E,
        bounds=(config.xmin, 1.0),
        initialize=lambda _, e: float(initial_density[int(e)]),
    )
    model.u = pyo.Var(
        model.F,
        initialize=lambda _, i: float(initial_free_displacement[int(i)]),
    )

    def rho_rule(m: pyo.ConcreteModel, e: int):
        return sum(weight * m.x[j] for j, weight in filter_data.rows[int(e)])

    model.rho = pyo.Expression(model.E, rule=rho_rule)

    def stiffness_rule(m: pyo.ConcreteModel, e: int):
        return config.emin + (config.e0 - config.emin) * m.rho[e] ** penalty

    model.stiffness_factor = pyo.Expression(model.E, rule=stiffness_rule)

    def equilibrium_rule(m: pyo.ConcreteModel, row: int):
        lhs = sum(
            value * m.stiffness_factor[element] * m.u[col]
            for element, col, value in row_terms[int(row)]
        )
        return lhs == float(mesh.force[mesh.free_dofs[int(row)]])

    model.equilibrium = pyo.Constraint(model.F, rule=equilibrium_rule)
    model.volume_fraction = pyo.Expression(
        expr=sum(model.rho[e] * mesh.element_area for e in model.E)
        / (mesh.element_count * mesh.element_area)
    )
    model.volume_constraint = pyo.Constraint(expr=model.volume_fraction <= config.volfrac)
    model.compliance = pyo.Expression(
        expr=sum(float(mesh.force[mesh.free_dofs[i]]) * model.u[i] for i in model.F)
    )
    model.objective = pyo.Objective(expr=model.compliance, sense=pyo.minimize)

    data = SimpModelData(
        mesh=mesh,
        element_ke=element_ke,
        filter_data=filter_data,
        row_terms=row_terms,
        config=config,
        penalty=penalty,
    )
    return model, data


def _read_finite_variable_vector(
    indexed_var: pyo.Var,
    indices,
    expected_size: int,
    name: str,
) -> np.ndarray:
    index_list = list(indices)
    if len(index_list) != expected_size:
        raise ValueError(f"{name} length must be {expected_size}")

    values = np.empty(expected_size, dtype=float)
    for position, index in enumerate(index_list):
        raw_value = pyo.value(indexed_var[index], exception=False)
        if raw_value is None:
            raise RuntimeError(f"{name}[{index}] has no value")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{name}[{index}] is not numeric") from exc
        if not np.isfinite(value):
            raise RuntimeError(f"{name}[{index}] must be finite")
        values[position] = value
    return values


def extract_solution(
    model: pyo.ConcreteModel,
    data: SimpModelData,
    stage_results: list[SimpStageResult],
    stage_snapshots: list[SimpStageSnapshot] | None = None,
) -> SimpSolution:
    raw_density = _read_finite_variable_vector(
        model.x,
        model.E,
        data.mesh.element_count,
        "raw density",
    )
    free_displacement = _read_finite_variable_vector(
        model.u,
        model.F,
        data.mesh.free_dofs.size,
        "free displacement",
    )
    physical_density = apply_density_filter(data.filter_data, raw_density)
    if not np.all(np.isfinite(physical_density)):
        raise RuntimeError("physical density must contain only finite values")

    full_displacement = np.zeros(data.mesh.total_dofs, dtype=float)
    full_displacement[data.mesh.free_dofs] = free_displacement

    return SimpSolution(
        raw_density=raw_density,
        physical_density=physical_density,
        free_displacement=free_displacement,
        full_displacement=full_displacement,
        stage_results=list(stage_results),
        model_data=data,
        stage_snapshots=list(stage_snapshots or []),
    )


def _build_stage_snapshot(solution: SimpSolution) -> SimpStageSnapshot:
    return SimpStageSnapshot(
        penalty=float(solution.model_data.penalty),
        raw_density=solution.raw_density.copy(),
        physical_density=solution.physical_density.copy(),
        free_displacement=solution.free_displacement.copy(),
        full_displacement=solution.full_displacement.copy(),
    )


def _read_finite_expression_value(expression, name: str) -> float:
    raw_value = pyo.value(expression, exception=False)
    if raw_value is None:
        raise RuntimeError(f"{name} has no value")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} is not numeric") from exc
    if not np.isfinite(value):
        raise RuntimeError(f"{name} must be finite")
    return value


def _read_expression_value_or_nan(expression) -> float:
    raw_value = pyo.value(expression, exception=False)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    return value


def _read_max_displacement_or_nan(
    model: pyo.ConcreteModel,
    data: SimpModelData,
) -> float:
    try:
        free_displacement = _read_finite_variable_vector(
            model.u,
            model.F,
            data.mesh.free_dofs.size,
            "free displacement",
        )
    except (RuntimeError, ValueError):
        return float("nan")
    full_displacement = np.zeros(data.mesh.total_dofs, dtype=float)
    full_displacement[data.mesh.free_dofs] = free_displacement
    return float(np.max(np.abs(full_displacement)))


def _build_stage_result(
    model: pyo.ConcreteModel,
    data: SimpModelData,
    solver_results,
    elapsed_seconds: float,
    solution: SimpSolution | None,
) -> SimpStageResult:
    if solution is None:
        compliance = _read_expression_value_or_nan(model.compliance)
        volume_fraction = _read_expression_value_or_nan(model.volume_fraction)
        max_displacement = _read_max_displacement_or_nan(model, data)
    else:
        compliance = _read_finite_expression_value(model.compliance, "compliance")
        volume_fraction = _read_finite_expression_value(
            model.volume_fraction,
            "volume_fraction",
        )
        max_displacement = float(np.max(np.abs(solution.full_displacement)))

    return SimpStageResult(
        penalty=float(data.penalty),
        compliance=compliance,
        volume_fraction=volume_fraction,
        max_displacement=max_displacement,
        solver_status=str(solver_results.solver.status),
        termination_condition=str(solver_results.solver.termination_condition),
        elapsed_seconds=float(elapsed_seconds),
    )


def solve_simp_continuation(
    config: SimpConfig,
    solver_settings: SolverSettings | None = None,
    tee: bool = False,
) -> SimpSolution:
    penalty_values = _validate_penalties(config.penalties)

    solver_settings = solver_settings or SolverSettings(tee=tee)
    solver_tee = bool(tee or solver_settings.tee)
    solver = create_ipopt_solver(solver_settings)

    stage_results: list[SimpStageResult] = []
    stage_snapshots: list[SimpStageSnapshot] = []
    initial_density: np.ndarray | None = None
    initial_free_displacement: np.ndarray | None = None
    final_solution: SimpSolution | None = None

    for penalty in penalty_values:
        model, data = build_simp_model(
            config=config,
            penalty=penalty,
            initial_density=initial_density,
            initial_free_displacement=initial_free_displacement,
        )

        started_at = time.perf_counter()
        results = solver.solve(model, tee=solver_tee)
        elapsed_seconds = time.perf_counter() - started_at

        if not check_optimal_termination(results):
            stage_results.append(
                _build_stage_result(
                    model=model,
                    data=data,
                    solver_results=results,
                    elapsed_seconds=elapsed_seconds,
                    solution=None,
                )
            )
            if final_solution is None:
                raise RuntimeError(
                    "SIMP continuation failed before any accepted solution: "
                    f"status={results.solver.status}, "
                    f"termination={results.solver.termination_condition}"
                )
            final_solution.stage_results = list(stage_results)
            final_solution.stage_snapshots = list(stage_snapshots)
            return final_solution

        solution = extract_solution(
            model,
            data,
            stage_results=stage_results,
            stage_snapshots=stage_snapshots,
        )
        stage_results.append(
            _build_stage_result(
                model=model,
                data=data,
                solver_results=results,
                elapsed_seconds=elapsed_seconds,
                solution=solution,
            )
        )
        stage_snapshots.append(_build_stage_snapshot(solution))
        solution.stage_results = list(stage_results)
        solution.stage_snapshots = list(stage_snapshots)
        final_solution = solution
        initial_density = solution.raw_density.copy()
        initial_free_displacement = solution.free_displacement.copy()

    if final_solution is None:
        raise RuntimeError("solve_simp_continuation did not build any stages")
    return final_solution
