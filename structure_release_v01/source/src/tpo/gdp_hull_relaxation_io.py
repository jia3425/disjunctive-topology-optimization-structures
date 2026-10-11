from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from tpo.gdp_hull_relaxation import (
    GDPHullRelaxationConfig,
    GDPHullRelaxationSolution,
    create_gdp_hull_relaxation_initial_solution,
)


_METRIC_FIELDS = [
    "method",
    "benchmark",
    "nelx",
    "nely",
    "element_count",
    "volfrac",
    "use_element_connectivity",
    "fix_load_elements",
    "phase_filter_radius",
    "heaviside_eta",
    "final_heaviside_beta",
    "final_solid_displacement_coupling_penalty",
    "final_solid_displacement_coupling",
    "final_solid_displacement_coupling_phase_law",
    "final_solid_displacement_coupling_phase_exponent",
    "final_solid_displacement_coupling_ramp_q",
    "final_solid_displacement_coupling_norm",
    "final_solid_displacement_coupling_mode",
    "use_tight_displacement_bounds",
    "tight_displacement_bound_safety_factor",
    "tight_displacement_bound_floor_fraction",
    "enforce_full_solid_compliance_lower_bound",
    "full_solid_compliance",
    "actual_volume_fraction",
    "continuous_compliance",
    "global_grayness",
    "max_element_grayness",
    "rounded_binary_compliance",
    "rounded_binary_volume_fraction",
    "rounded_response_status",
    "stage_count",
    "completed_all_stages",
    "failed_stage_index",
    "failed_stage_reason",
    "elapsed_seconds",
]


def _path(output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finite(value, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def build_gdp_hull_relaxation_summary(solution: GDPHullRelaxationSolution) -> dict[str, object]:
    mesh = solution.model_data.mesh
    config = solution.model_data.config
    elapsed = float(sum(stage.elapsed_seconds for stage in solution.stage_results))
    return {
        "method": "GDP-Hull-Relaxation",
        "benchmark": str(config.benchmark),
        "nelx": int(mesh.nelx),
        "nely": int(mesh.nely),
        "length": float(mesh.length),
        "height": float(mesh.height),
        "element_count": int(mesh.element_count),
        "volfrac": float(config.volfrac),
        "use_element_connectivity": bool(config.use_element_connectivity),
        "fix_load_elements": bool(config.fix_load_elements),
        "phase_filter_radius": config.phase_filter_radius,
        "heaviside_eta": float(config.heaviside_eta),
        "final_heaviside_beta": None
        if not solution.stage_results
        else solution.stage_results[-1].heaviside_beta,
        "final_solid_displacement_coupling_penalty": None
        if not solution.stage_results
        else solution.stage_results[-1].solid_displacement_coupling_penalty,
        "final_solid_displacement_coupling": None
        if not solution.stage_results
        else solution.stage_results[-1].solid_displacement_coupling,
        "final_solid_displacement_coupling_phase_law": None
        if not solution.stage_results
        else solution.stage_results[-1].solid_displacement_coupling_phase_law,
        "final_solid_displacement_coupling_phase_exponent": None
        if not solution.stage_results
        else solution.stage_results[-1].solid_displacement_coupling_phase_exponent,
        "final_solid_displacement_coupling_ramp_q": None
        if not solution.stage_results
        else solution.stage_results[-1].solid_displacement_coupling_ramp_q,
        "final_solid_displacement_coupling_norm": None
        if not solution.stage_results
        else solution.stage_results[-1].solid_displacement_coupling_norm,
        "final_solid_displacement_coupling_mode": None
        if not solution.stage_results
        else solution.stage_results[-1].solid_displacement_coupling_mode,
        "use_tight_displacement_bounds": bool(config.use_tight_displacement_bounds),
        "tight_displacement_bound_safety_factor": float(config.tight_displacement_bound_safety_factor),
        "tight_displacement_bound_floor_fraction": float(config.tight_displacement_bound_floor_fraction),
        "enforce_full_solid_compliance_lower_bound": bool(
            config.enforce_full_solid_compliance_lower_bound
        ),
        "full_solid_compliance": _finite(
            solution.model_data.full_solid_compliance,
            "full_solid_compliance",
        ),
        "actual_volume_fraction": _finite(solution.volume_fraction, "actual_volume_fraction"),
        "continuous_compliance": _finite(solution.continuous_compliance, "continuous_compliance"),
        "global_grayness": _finite(solution.global_grayness, "global_grayness"),
        "max_element_grayness": _finite(solution.max_element_grayness, "max_element_grayness"),
        "rounded_binary_compliance": None
        if solution.rounded_binary_compliance is None
        else _finite(solution.rounded_binary_compliance, "rounded_binary_compliance"),
        "rounded_binary_volume_fraction": _finite(
            solution.rounded_binary_volume_fraction,
            "rounded_binary_volume_fraction",
        ),
        "rounded_response_status": str(solution.rounded_response_status),
        "stage_count": int(len(solution.stage_results)),
        "completed_all_stages": bool(solution.completed_all_stages),
        "failed_stage_index": solution.failed_stage_index,
        "failed_stage_reason": solution.failed_stage_reason,
        "elapsed_seconds": elapsed,
        "python_executable": sys.executable,
    }


def write_gdp_hull_relaxation_summary_json(
    solution: GDPHullRelaxationSolution,
    output_path: Path,
) -> None:
    with _path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(build_gdp_hull_relaxation_summary(solution), handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_gdp_hull_relaxation_metrics_summary_csv(
    solution: GDPHullRelaxationSolution,
    output_path: Path,
) -> None:
    summary = build_gdp_hull_relaxation_summary(solution)
    with _path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_METRIC_FIELDS)
        writer.writeheader()
        writer.writerow({field: summary.get(field, "") for field in _METRIC_FIELDS})


def write_gdp_hull_relaxation_stage_history_csv(
    solution: GDPHullRelaxationSolution,
    output_path: Path,
) -> None:
    fields = [
        "stage_index",
        "penalty",
        "grayness_limit",
        "heaviside_beta",
        "compliance",
        "normalized_objective",
        "global_grayness",
        "max_element_grayness",
        "volume_fraction",
        "solid_displacement_coupling_penalty",
        "solid_displacement_coupling",
        "solid_displacement_coupling_phase_law",
        "solid_displacement_coupling_phase_exponent",
        "solid_displacement_coupling_ramp_q",
        "solid_displacement_coupling_norm",
        "solid_displacement_coupling_mode",
        "solver_status",
        "termination_condition",
        "elapsed_seconds",
    ]
    with _path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in solution.stage_results:
            writer.writerow({field: getattr(stage, field) for field in fields})


def write_gdp_hull_relaxation_solution_npz(
    solution: GDPHullRelaxationSolution,
    output_path: Path,
) -> None:
    mesh = solution.model_data.mesh
    rounded_displacement = (
        np.full(mesh.total_dofs, np.nan, dtype=float)
        if solution.rounded_binary_displacement is None
        else solution.rounded_binary_displacement
    )
    np.savez_compressed(
        _path(output_path),
        nodes=mesh.nodes,
        elements=mesh.elements,
        force=mesh.force,
        element_dofs=mesh.element_dofs,
        fixed_dofs=mesh.fixed_dofs,
        free_dofs=mesh.free_dofs,
        ys=solution.ys,
        yv=solution.yv,
        raw_ys=np.full(mesh.element_count, np.nan, dtype=float) if solution.raw_ys is None else solution.raw_ys,
        raw_yv=np.full(mesh.element_count, np.nan, dtype=float) if solution.raw_yv is None else solution.raw_yv,
        filtered_ys=np.full(mesh.element_count, np.nan, dtype=float)
        if solution.filtered_ys is None
        else solution.filtered_ys,
        displacement=solution.full_displacement,
        solid_displacement_components=solution.solid_displacement_components,
        full_solid_displacement=solution.model_data.full_solid_displacement,
        displacement_bounds=solution.model_data.displacement_bounds,
        rounded_binary_density=solution.rounded_binary_density,
        rounded_binary_displacement=rounded_displacement,
    )


def load_gdp_hull_relaxation_initial_solution_npz(
    config: GDPHullRelaxationConfig,
    solution_path: Path,
    density_field: str = "raw_ys",
) -> GDPHullRelaxationSolution:
    path = _path(solution_path)
    if density_field not in {"ys", "raw_ys", "filtered_ys"}:
        raise ValueError("density_field must be ys, raw_ys, or filtered_ys")
    with np.load(path) as archive:
        if density_field not in archive.files:
            raise ValueError(f"{density_field} is missing from {path}")
        density = np.asarray(archive[density_field], dtype=float)
        raw_ys = np.asarray(archive["raw_ys"], dtype=float) if "raw_ys" in archive.files else density
        raw_yv = np.asarray(archive["raw_yv"], dtype=float) if "raw_yv" in archive.files else 1.0 - raw_ys
        filtered_ys = (
            np.asarray(archive["filtered_ys"], dtype=float)
            if "filtered_ys" in archive.files
            else None
        )
        full_displacement = (
            np.asarray(archive["displacement"], dtype=float)
            if "displacement" in archive.files
            else None
        )
        solid_components = (
            np.asarray(archive["solid_displacement_components"], dtype=float)
            if "solid_displacement_components" in archive.files
            else None
        )
    return create_gdp_hull_relaxation_initial_solution(
        config,
        density,
        full_displacement=full_displacement,
        solid_displacement_components=solid_components,
        raw_ys=raw_ys,
        raw_yv=raw_yv,
        filtered_ys=filtered_ys,
    )


__all__ = [
    "build_gdp_hull_relaxation_summary",
    "load_gdp_hull_relaxation_initial_solution_npz",
    "write_gdp_hull_relaxation_metrics_summary_csv",
    "write_gdp_hull_relaxation_solution_npz",
    "write_gdp_hull_relaxation_stage_history_csv",
    "write_gdp_hull_relaxation_summary_json",
]
