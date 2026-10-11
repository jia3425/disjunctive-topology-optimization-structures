from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from tpo.gdp_hull_relaxation import GDPHullRelaxationSolution


_DIAGNOSTIC_FIELDS = [
    "method",
    "benchmark",
    "nelx",
    "nely",
    "element_count",
    "volfrac",
    "continuous_compliance",
    "full_solid_compliance",
    "compliance_to_full_solid_ratio",
    "global_grayness",
    "max_element_grayness",
    "projected_density_min",
    "projected_density_mean",
    "projected_density_std",
    "projected_density_max",
    "projected_density_q05",
    "projected_density_q50",
    "projected_density_q95",
    "raw_density_std",
    "filtered_density_std",
    "u_vs_full_solid_relative_l2",
    "u_vs_full_solid_correlation",
    "max_abs_displacement",
    "max_abs_full_solid_displacement",
    "solid_displacement_coupling_relative_l2",
    "solid_displacement_coupling_mean_abs",
    "solid_displacement_coupling_max_abs",
    "solid_bound_activity_mean",
    "solid_bound_activity_q95",
    "solid_bound_activity_max",
    "rounded_solid_count",
    "rounded_component_count",
    "rounded_largest_component_size",
    "rounded_support_load_connected",
    "rounded_support_solid_count",
    "rounded_load_solid_count",
    "rounded_binary_compliance",
    "rounded_response_status",
]


def _path(output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _relative_l2(values: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(values, dtype=float) - np.asarray(reference, dtype=float)))
    denominator = float(np.linalg.norm(np.asarray(reference, dtype=float)))
    if denominator <= 1e-12:
        denominator = max(float(np.linalg.norm(np.asarray(values, dtype=float))), 1.0)
    return numerator / denominator


def _correlation(values: np.ndarray, reference: np.ndarray) -> float | None:
    vector = np.asarray(values, dtype=float).ravel()
    baseline = np.asarray(reference, dtype=float).ravel()
    if vector.size != baseline.size or vector.size < 2:
        return None
    if float(np.std(vector)) <= 1e-12 or float(np.std(baseline)) <= 1e-12:
        return None
    return _finite_or_none(float(np.corrcoef(vector, baseline)[0, 1]))


def _density_statistics(prefix: str, values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=float).ravel()
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{prefix} density values must be nonempty and finite")
    q05, q50, q95 = np.quantile(vector, [0.05, 0.5, 0.95])
    return {
        f"{prefix}_density_min": float(np.min(vector)),
        f"{prefix}_density_mean": float(np.mean(vector)),
        f"{prefix}_density_std": float(np.std(vector)),
        f"{prefix}_density_max": float(np.max(vector)),
        f"{prefix}_density_q05": float(q05),
        f"{prefix}_density_q50": float(q50),
        f"{prefix}_density_q95": float(q95),
    }


def _solid_phase_stiffness(solution: GDPHullRelaxationSolution) -> np.ndarray:
    density = np.asarray(solution.ys, dtype=float)
    if not solution.stage_results:
        return density
    stage = solution.stage_results[-1]
    law = str(getattr(stage, "solid_displacement_coupling_phase_law", "linear")).strip().lower()
    if law == "simp":
        exponent = float(getattr(stage, "solid_displacement_coupling_phase_exponent", 1.0))
        return np.power(np.clip(density, 0.0, 1.0), exponent)
    if law == "ramp":
        q = float(getattr(stage, "solid_displacement_coupling_ramp_q", 8.0))
        return density / (1.0 + q * (1.0 - density))
    return density


def _rounded_connectivity_metrics(solution: GDPHullRelaxationSolution) -> dict[str, Any]:
    data = solution.model_data
    density = np.asarray(solution.rounded_binary_density, dtype=float).ravel()
    solid_elements = set(int(index) for index in np.flatnonzero(density > 0.5))
    support_elements = set(int(index) for index in data.support_elements)
    load_elements = set(int(index) for index in data.load_elements)
    if not solid_elements:
        return {
            "rounded_solid_count": 0,
            "rounded_component_count": 0,
            "rounded_largest_component_size": 0,
            "rounded_support_load_connected": False,
            "rounded_support_solid_count": 0,
            "rounded_load_solid_count": 0,
        }

    adjacency = {element: set() for element in range(data.mesh.element_count)}
    for source, target in data.element_directed_edges:
        adjacency[int(source)].add(int(target))
        adjacency[int(target)].add(int(source))

    component_count = 0
    largest_component = 0
    support_load_connected = False
    visited: set[int] = set()
    for start in sorted(solid_elements):
        if start in visited:
            continue
        component_count += 1
        queue: deque[int] = deque([start])
        visited.add(start)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in solid_elements and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        component_set = set(component)
        largest_component = max(largest_component, len(component))
        if component_set & support_elements and component_set & load_elements:
            support_load_connected = True

    return {
        "rounded_solid_count": int(len(solid_elements)),
        "rounded_component_count": int(component_count),
        "rounded_largest_component_size": int(largest_component),
        "rounded_support_load_connected": bool(support_load_connected),
        "rounded_support_solid_count": int(len(solid_elements & support_elements)),
        "rounded_load_solid_count": int(len(solid_elements & load_elements)),
    }


def build_gdp_hull_relaxation_diagnostics(solution: GDPHullRelaxationSolution) -> dict[str, Any]:
    data = solution.model_data
    mesh = data.mesh
    diagnostics: dict[str, Any] = {
        "method": "GDP-Hull-Relaxation-Diagnostics",
        "benchmark": str(data.config.benchmark),
        "nelx": int(mesh.nelx),
        "nely": int(mesh.nely),
        "element_count": int(mesh.element_count),
        "volfrac": float(data.config.volfrac),
        "continuous_compliance": float(solution.continuous_compliance),
        "full_solid_compliance": float(data.full_solid_compliance),
        "compliance_to_full_solid_ratio": None
        if data.full_solid_compliance <= 0.0
        else float(solution.continuous_compliance / data.full_solid_compliance),
        "global_grayness": float(solution.global_grayness),
        "max_element_grayness": float(solution.max_element_grayness),
        "rounded_binary_compliance": solution.rounded_binary_compliance,
        "rounded_response_status": str(solution.rounded_response_status),
    }
    diagnostics.update(_density_statistics("projected", solution.ys))
    if solution.raw_ys is not None:
        diagnostics.update(_density_statistics("raw", solution.raw_ys))
    if solution.filtered_ys is not None:
        diagnostics.update(_density_statistics("filtered", solution.filtered_ys))

    free_dofs = np.asarray(mesh.free_dofs, dtype=int)
    diagnostics["u_vs_full_solid_relative_l2"] = _relative_l2(
        solution.full_displacement[free_dofs],
        data.full_solid_displacement[free_dofs],
    )
    diagnostics["u_vs_full_solid_correlation"] = _correlation(
        solution.full_displacement[free_dofs],
        data.full_solid_displacement[free_dofs],
    )
    diagnostics["max_abs_displacement"] = float(np.max(np.abs(solution.full_displacement)))
    diagnostics["max_abs_full_solid_displacement"] = float(np.max(np.abs(data.full_solid_displacement)))

    element_global_displacement = solution.full_displacement[mesh.element_dofs]
    solid_phase = _solid_phase_stiffness(solution).reshape((-1, 1))
    solid_components = np.asarray(solution.solid_displacement_components, dtype=float)
    if solid_components.shape != element_global_displacement.shape:
        raise ValueError("solid_displacement_components shape does not match element displacement map")
    coupling_residual = solid_components - solid_phase * element_global_displacement
    diagnostics["solid_displacement_coupling_relative_l2"] = _relative_l2(
        solid_components,
        solid_phase * element_global_displacement,
    )
    diagnostics["solid_displacement_coupling_mean_abs"] = float(np.mean(np.abs(coupling_residual)))
    diagnostics["solid_displacement_coupling_max_abs"] = float(np.max(np.abs(coupling_residual)))

    solid_capacity = data.displacement_bounds[mesh.element_dofs] * np.maximum(solid_phase, 1e-12)
    solid_activity = np.abs(solid_components) / np.maximum(solid_capacity, 1e-12)
    diagnostics["solid_bound_activity_mean"] = float(np.mean(solid_activity))
    diagnostics["solid_bound_activity_q95"] = float(np.quantile(solid_activity, 0.95))
    diagnostics["solid_bound_activity_max"] = float(np.max(solid_activity))
    diagnostics.update(_rounded_connectivity_metrics(solution))
    return diagnostics


def write_gdp_hull_relaxation_diagnostics_json(
    solution: GDPHullRelaxationSolution,
    output_path: Path,
) -> None:
    with _path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(build_gdp_hull_relaxation_diagnostics(solution), handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_gdp_hull_relaxation_diagnostics_csv(
    solution: GDPHullRelaxationSolution,
    output_path: Path,
) -> None:
    diagnostics = build_gdp_hull_relaxation_diagnostics(solution)
    fields = list(_DIAGNOSTIC_FIELDS)
    for key in diagnostics:
        if key not in fields:
            fields.append(key)
    with _path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: diagnostics.get(field, "") for field in fields})


__all__ = [
    "build_gdp_hull_relaxation_diagnostics",
    "write_gdp_hull_relaxation_diagnostics_csv",
    "write_gdp_hull_relaxation_diagnostics_json",
]
