from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tpo.mesh import Mesh, element_values_to_grid

try:
    from pyparsing import PyparsingDeprecationWarning
except ImportError:  # pragma: no cover - depends on pyparsing version
    PyparsingDeprecationWarning = None

if PyparsingDeprecationWarning is not None:
    warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"matplotlib\..*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"pyparsing\..*")


def _finite_vector(values, expected_size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite NumPy-compatible array") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size != expected_size:
        raise ValueError(f"{name} length must be {expected_size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _field_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")
    return cmap


def _field_figsize(mesh: Mesh) -> tuple[float, float]:
    aspect = mesh.length / mesh.height
    if aspect >= 1.0:
        return (max(5.0, min(9.0, 3.0 * aspect)), 3.4)
    return (4.2, max(4.2, min(7.0, 3.6 / max(aspect, 1e-6))))


def save_density_plot(
    mesh: Mesh,
    density: np.ndarray,
    output_path: Path,
    title: str = "SIMP physical density",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field = element_values_to_grid(
        mesh,
        density,
        name="density",
    )

    fig, ax = plt.subplots(figsize=_field_figsize(mesh))
    try:
        image = ax.imshow(
            np.ma.masked_invalid(field),
            cmap=_field_cmap("gray_r"),
            extent=(0.0, mesh.length, 0.0, mesh.height),
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_title(str(title))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def save_displacement_plot(
    mesh: Mesh,
    displacement: np.ndarray,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodal_displacement = _finite_vector(
        displacement,
        mesh.total_dofs,
        "displacement",
    ).reshape((-1, 2))
    magnitude = np.linalg.norm(nodal_displacement, axis=1).reshape(
        mesh.nely + 1,
        mesh.nelx + 1,
    )
    if mesh.elements.size:
        active_nodes = np.unique(mesh.elements.ravel())
        node_mask = np.zeros(mesh.nodes.shape[0], dtype=bool)
        node_mask[active_nodes] = True
        magnitude = magnitude.astype(float, copy=True)
        magnitude[~node_mask.reshape(mesh.nely + 1, mesh.nelx + 1)] = np.nan

    fig, ax = plt.subplots(figsize=_field_figsize(mesh))
    try:
        image = ax.imshow(
            np.ma.masked_invalid(magnitude),
            cmap=_field_cmap("viridis"),
            extent=(0.0, mesh.length, 0.0, mesh.height),
            origin="lower",
            aspect="auto",
        )
        ax.set_title("Displacement magnitude")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def _stage_penalties(stage_items: Iterable) -> np.ndarray:
    penalties = np.asarray([float(item.penalty) for item in stage_items], dtype=float)
    if penalties.ndim != 1 or penalties.size == 0:
        raise ValueError("stage data must contain at least one stage")
    if not np.all(np.isfinite(penalties)):
        raise ValueError("stage penalties must contain only finite values")
    return penalties


def _stage_values(stage_items: Iterable, attribute: str, name: str) -> np.ndarray:
    values = np.asarray(
        [float(getattr(item, attribute)) for item in stage_items],
        dtype=float,
    )
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _iteration_indices(iteration_items: Iterable) -> np.ndarray:
    iterations = np.asarray([int(item.iteration) for item in iteration_items], dtype=int)
    if iterations.ndim != 1 or iterations.size == 0:
        raise ValueError("iteration data must contain at least one item")
    if np.any(iterations <= 0):
        raise ValueError("iteration indices must be positive")
    return iterations


def _iteration_values(iteration_items: Iterable, attribute: str, name: str) -> np.ndarray:
    values = np.asarray(
        [float(getattr(item, attribute)) for item in iteration_items],
        dtype=float,
    )
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def save_density_evolution_plot(mesh: Mesh, stage_snapshots: list, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not stage_snapshots:
        raise ValueError("stage_snapshots must contain at least one accepted stage")

    fields = [
        element_values_to_grid(
            mesh,
            snapshot.physical_density,
            name=f"stage {index} physical_density",
        )
        for index, snapshot in enumerate(stage_snapshots)
    ]
    fig_width = max(4.0, 3.2 * len(fields))
    fig, axes = plt.subplots(1, len(fields), figsize=(fig_width, 3.4), squeeze=False)
    try:
        image = None
        for ax, snapshot, field in zip(axes.ravel(), stage_snapshots, fields):
            image = ax.imshow(
                np.ma.masked_invalid(field),
                cmap=_field_cmap("gray_r"),
                extent=(0.0, mesh.length, 0.0, mesh.height),
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
            )
            ax.set_title(f"p = {float(snapshot.penalty):g}")
            ax.set_xlabel("x")
            if ax is axes.ravel()[0]:
                ax.set_ylabel("y")
        fig.suptitle("SIMP density evolution", y=0.98)
        fig.subplots_adjust(top=0.78, right=0.88, wspace=0.28)
        if image is not None:
            colorbar_ax = fig.add_axes([0.91, 0.18, 0.025, 0.58])
            fig.colorbar(image, cax=colorbar_ax)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def save_iteration_density_evolution_plot(
    mesh: Mesh,
    iteration_snapshots: list,
    output_path: Path,
    max_panels: int = 6,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not iteration_snapshots:
        raise ValueError("iteration_snapshots must contain at least one snapshot")
    if max_panels <= 0:
        raise ValueError("max_panels must be positive")

    if len(iteration_snapshots) > max_panels:
        selected_indices = np.linspace(0, len(iteration_snapshots) - 1, max_panels, dtype=int)
        snapshots = [iteration_snapshots[int(index)] for index in selected_indices]
    else:
        snapshots = list(iteration_snapshots)

    fields = [
        element_values_to_grid(
            mesh,
            snapshot.physical_density,
            name=f"iteration {snapshot.iteration} physical_density",
        )
        for snapshot in snapshots
    ]
    fig_width = max(4.0, 3.0 * len(fields))
    fig, axes = plt.subplots(1, len(fields), figsize=(fig_width, 3.4), squeeze=False)
    try:
        image = None
        for ax, snapshot, field in zip(axes.ravel(), snapshots, fields):
            image = ax.imshow(
                np.ma.masked_invalid(field),
                cmap=_field_cmap("gray_r"),
                extent=(0.0, mesh.length, 0.0, mesh.height),
                origin="lower",
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
            )
            ax.set_title(f"iter = {int(snapshot.iteration)}")
            ax.set_xlabel("x")
            if ax is axes.ravel()[0]:
                ax.set_ylabel("y")
        fig.suptitle("OC-SIMP density evolution", y=0.98)
        fig.subplots_adjust(top=0.78, right=0.88, wspace=0.28)
        if image is not None:
            colorbar_ax = fig.add_axes([0.91, 0.18, 0.025, 0.58])
            fig.colorbar(image, cax=colorbar_ax)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def _save_stage_line_plot(
    stage_results: list,
    output_path: Path,
    attribute: str,
    value_name: str,
    ylabel: str,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    penalties = _stage_penalties(stage_results)
    values = _stage_values(stage_results, attribute, value_name)

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.plot(penalties, values, marker="o", linewidth=2.0)
        ax.set_xlabel("SIMP penalty p")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def _save_iteration_line_plot(
    iteration_results: list,
    output_path: Path,
    attribute: str,
    value_name: str,
    ylabel: str,
    title: str,
    log_y: bool = False,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = _iteration_indices(iteration_results)
    values = _iteration_values(iteration_results, attribute, value_name)

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.plot(iterations, values, linewidth=1.8)
        ax.set_xlabel("OC iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def save_objective_history_plot(stage_results: list, output_path: Path) -> None:
    _save_stage_line_plot(
        stage_results,
        output_path,
        attribute="compliance",
        value_name="compliance",
        ylabel="Compliance",
        title="Objective history",
    )


def save_oc_objective_history_plot(iteration_results: list, output_path: Path) -> None:
    _save_iteration_line_plot(
        iteration_results,
        output_path,
        attribute="compliance",
        value_name="compliance",
        ylabel="Compliance",
        title="OC-SIMP objective history",
    )


def save_volume_history_plot(stage_results: list, output_path: Path) -> None:
    _save_stage_line_plot(
        stage_results,
        output_path,
        attribute="volume_fraction",
        value_name="volume fraction",
        ylabel="Volume fraction",
        title="Volume history",
    )


def save_oc_volume_history_plot(iteration_results: list, output_path: Path) -> None:
    _save_iteration_line_plot(
        iteration_results,
        output_path,
        attribute="volume_fraction",
        value_name="volume fraction",
        ylabel="Volume fraction",
        title="OC-SIMP volume history",
    )


def save_oc_change_history_plot(iteration_results: list, output_path: Path) -> None:
    _save_iteration_line_plot(
        iteration_results,
        output_path,
        attribute="change",
        value_name="density change",
        ylabel="Max density change",
        title="OC-SIMP change history",
        log_y=True,
    )


def save_time_history_plot(stage_results: list, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    penalties = _stage_penalties(stage_results)
    elapsed = _stage_values(stage_results, "elapsed_seconds", "elapsed seconds")
    if np.any(elapsed < 0.0):
        raise ValueError("elapsed seconds must be nonnegative")
    cumulative = np.cumsum(elapsed)

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.bar(penalties, elapsed, width=0.35, alpha=0.65, label="stage")
        ax.plot(penalties, cumulative, marker="o", color="black", label="cumulative")
        ax.set_xlabel("SIMP penalty p")
        ax.set_ylabel("Seconds")
        ax.set_title("Solve time history")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


def save_oc_time_history_plot(iteration_results: list, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = _iteration_indices(iteration_results)
    cumulative = _iteration_values(
        iteration_results,
        "cumulative_elapsed_seconds",
        "cumulative elapsed seconds",
    )
    if np.any(cumulative < 0.0):
        raise ValueError("cumulative elapsed seconds must be nonnegative")

    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.plot(iterations, cumulative, linewidth=1.8)
        ax.set_xlabel("OC iteration")
        ax.set_ylabel("Cumulative seconds")
        ax.set_title("OC-SIMP solve time history")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
    finally:
        plt.close(fig)


__all__ = [
    "save_density_evolution_plot",
    "save_density_plot",
    "save_displacement_plot",
    "save_iteration_density_evolution_plot",
    "save_oc_change_history_plot",
    "save_oc_objective_history_plot",
    "save_oc_time_history_plot",
    "save_oc_volume_history_plot",
    "save_objective_history_plot",
    "save_time_history_plot",
    "save_volume_history_plot",
]
