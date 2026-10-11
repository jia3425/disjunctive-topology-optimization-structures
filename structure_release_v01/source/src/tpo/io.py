from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tpo.runtime import IPOPT_EXECUTABLE

if TYPE_CHECKING:
    from tpo.simp_model import SimpSolution


_CONVERGENCE_FIELDS = [
    "penalty",
    "compliance",
    "volume_fraction",
    "max_displacement",
    "solver_status",
    "termination_condition",
    "elapsed_seconds",
]
_METRICS_FIELDS = [
    "stage_index",
    "accepted",
    *_CONVERGENCE_FIELDS,
    "cumulative_elapsed_seconds",
]


def _output_path(output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finite_float(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_array(values, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite NumPy-compatible array") from exc
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integer_array(values, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a NumPy-compatible integer array") from exc
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape must be {expected_shape}")
    return array


def write_convergence_csv(solution: "SimpSolution", output_path: Path) -> None:
    path = _output_path(output_path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CONVERGENCE_FIELDS)
        writer.writeheader()
        for stage in solution.stage_results:
            writer.writerow({field: getattr(stage, field) for field in _CONVERGENCE_FIELDS})


def _require_stage_results(solution: SimpSolution):
    if not solution.stage_results:
        raise ValueError("solution.stage_results must contain at least one stage result")
    return solution.stage_results


def _is_accepted_stage(stage) -> bool:
    return (
        str(stage.solver_status) == "ok"
        and str(stage.termination_condition)
        in {"optimal", "locallyOptimal", "globallyOptimal"}
    )


def _last_accepted_stage(stage_results):
    for stage in reversed(stage_results):
        if _is_accepted_stage(stage):
            return stage
    raise ValueError("solution.stage_results must contain at least one accepted stage")


def write_metrics_summary_csv(solution: "SimpSolution", output_path: Path) -> None:
    stage_results = _require_stage_results(solution)
    path = _output_path(output_path)
    cumulative_elapsed = 0.0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_METRICS_FIELDS)
        writer.writeheader()
        for index, stage in enumerate(stage_results, start=1):
            elapsed = _finite_float(
                stage.elapsed_seconds,
                f"stage {index} elapsed_seconds",
            )
            cumulative_elapsed += elapsed
            row = {field: getattr(stage, field) for field in _CONVERGENCE_FIELDS}
            row.update(
                {
                    "stage_index": index,
                    "accepted": _is_accepted_stage(stage),
                    "cumulative_elapsed_seconds": cumulative_elapsed,
                }
            )
            writer.writerow(row)


def build_summary(solution: "SimpSolution") -> dict[str, object]:
    stage_results = _require_stage_results(solution)
    data = solution.model_data
    final_stage = _last_accepted_stage(stage_results)
    elapsed_seconds = sum(
        _finite_float(stage.elapsed_seconds, f"stage {index} elapsed_seconds")
        for index, stage in enumerate(stage_results)
    )
    return {
        "method": "SIMP-Pyomo-IPOPT",
        "benchmark": str(data.config.benchmark),
        "nelx": int(data.mesh.nelx),
        "nely": int(data.mesh.nely),
        "length": float(data.mesh.length),
        "height": float(data.mesh.height),
        "element_count": int(data.mesh.element_count),
        "volfrac": float(data.config.volfrac),
        "rmin": float(data.config.rmin),
        "penalty_final": _finite_float(final_stage.penalty, "penalty_final"),
        "compliance": _finite_float(final_stage.compliance, "compliance"),
        "actual_volume_fraction": _finite_float(
            final_stage.volume_fraction,
            "actual_volume_fraction",
        ),
        "max_displacement": _finite_float(
            final_stage.max_displacement,
            "max_displacement",
        ),
        "solver_name": "ipopt",
        "solver_status": str(final_stage.solver_status),
        "termination_condition": str(final_stage.termination_condition),
        "elapsed_seconds": elapsed_seconds,
        "python_executable": sys.executable,
        "ipopt_executable": str(IPOPT_EXECUTABLE),
    }


def write_summary_json(solution: "SimpSolution", output_path: Path) -> None:
    path = _output_path(output_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(build_summary(solution), handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_solution_npz(solution: "SimpSolution", output_path: Path) -> None:
    path = _output_path(output_path)
    mesh = solution.model_data.mesh
    nodes = _finite_array(mesh.nodes, (mesh.nodes.shape[0], 2), "nodes")
    elements = _integer_array(mesh.elements, (mesh.element_count, 4), "elements")
    element_dofs = _integer_array(
        mesh.element_dofs,
        (mesh.element_count, 8),
        "element_dofs",
    )
    fixed_dofs = _integer_array(mesh.fixed_dofs, (mesh.fixed_dofs.size,), "fixed_dofs")
    free_dofs = _integer_array(mesh.free_dofs, (mesh.free_dofs.size,), "free_dofs")
    force = _finite_array(mesh.force, (mesh.total_dofs,), "force")
    if mesh.element_grid_indices is None:
        if mesh.element_count != mesh.nelx * mesh.nely:
            raise ValueError("mesh.element_grid_indices is required for inactive domains")
        grid_rows = np.repeat(np.arange(mesh.nely, dtype=int), mesh.nelx)
        grid_cols = np.tile(np.arange(mesh.nelx, dtype=int), mesh.nely)
        element_grid_indices = np.column_stack((grid_rows, grid_cols))
    else:
        element_grid_indices = _integer_array(
            mesh.element_grid_indices,
            (mesh.element_count, 2),
            "element_grid_indices",
        )
    if mesh.active_element_mask is None:
        active_element_mask = np.ones((mesh.nely, mesh.nelx), dtype=bool)
    else:
        active_element_mask = np.asarray(mesh.active_element_mask, dtype=bool)
        if active_element_mask.shape != (mesh.nely, mesh.nelx):
            raise ValueError(f"active_element_mask shape must be {(mesh.nely, mesh.nelx)}")
    raw_density = _finite_array(
        solution.raw_density,
        (mesh.element_count,),
        "raw_density",
    )
    physical_density = _finite_array(
        solution.physical_density,
        (mesh.element_count,),
        "physical_density",
    )
    displacement = _finite_array(
        solution.full_displacement,
        (mesh.total_dofs,),
        "displacement",
    )
    np.savez_compressed(
        path,
        nodes=nodes,
        elements=elements,
        element_dofs=element_dofs,
        element_grid_indices=element_grid_indices,
        active_element_mask=active_element_mask,
        fixed_dofs=fixed_dofs,
        free_dofs=free_dofs,
        force=force,
        raw_density=raw_density,
        physical_density=physical_density,
        displacement=displacement,
    )


__all__ = [
    "build_summary",
    "write_convergence_csv",
    "write_metrics_summary_csv",
    "write_solution_npz",
    "write_summary_json",
]
