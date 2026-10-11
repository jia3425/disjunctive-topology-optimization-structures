from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"D:\Case\TPO")
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def canonical_benchmark_label(benchmark: str) -> str:
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


def result_directory_name(benchmark: str, nelx: int, nely: int) -> str:
    return f"gdp_hull_relaxation_projected_weak_grayness_{canonical_benchmark_label(benchmark)}_{int(nelx)}x{int(nely)}"


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def _bounded_fraction(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not (0.0 < result <= 1.0):
        raise argparse.ArgumentTypeError("must satisfy 0 < value <= 1")
    return result


def _open_unit_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not (0.0 < result < 1.0):
        raise argparse.ArgumentTypeError("must satisfy 0 < value < 1")
    return result


def _parse_float_list(values: str, name: str) -> list[float]:
    parsed = [float(item.strip()) for item in str(values).split(",") if item.strip()]
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    for value in parsed:
        if not math.isfinite(value):
            raise ValueError(f"{name} values must be finite")
    return parsed


def _stagewise_positive_values(values: str | None, fallback: float, count: int, name: str) -> list[float]:
    if values is None:
        parsed = [float(fallback)]
    else:
        parsed = _parse_float_list(values, name)
    for value in parsed:
        if value <= 0.0:
            raise ValueError(f"{name} values must be positive")
    if len(parsed) == 1:
        return parsed * count
    if len(parsed) != count:
        raise ValueError(f"{name} must contain one value or have the same length as betas")
    return parsed


def parse_projected_grayness_stages(
    betas: str,
    grayness_limits: str,
    penalty: float = 0.0,
    coupling_penalties: str = "0",
    coupling_phase_law: str = "linear",
    coupling_phase_exponent: float = 1.0,
    coupling_phase_exponents: str | None = None,
    coupling_ramp_q: float = 8.0,
    coupling_norm: str = "bound",
    coupling_mode: str = "penalty",
):
    from tpo.gdp_hull_relaxation import GDPHullRelaxationStage

    beta_values = _parse_float_list(betas, "betas")
    limit_values = _parse_float_list(grayness_limits, "grayness_limits")
    if len(beta_values) != len(limit_values):
        raise ValueError("betas and grayness_limits must have the same length")
    for beta in beta_values:
        if beta <= 0.0:
            raise ValueError("betas must be positive")
    for limit in limit_values:
        if not (0.0 <= limit <= 1.0):
            raise ValueError("grayness_limits must satisfy 0 <= value <= 1")
    if not math.isfinite(float(penalty)) or float(penalty) < 0.0:
        raise ValueError("penalty must be nonnegative and finite")
    coupling_values = _parse_float_list(coupling_penalties, "coupling_penalties")
    if len(coupling_values) == 1:
        coupling_values = coupling_values * len(beta_values)
    elif len(coupling_values) != len(beta_values):
        raise ValueError("coupling_penalties must contain one value or have the same length as betas")
    for value in coupling_values:
        if value < 0.0:
            raise ValueError("coupling_penalties must be nonnegative")
    exponent_values = _stagewise_positive_values(
        coupling_phase_exponents,
        coupling_phase_exponent,
        len(beta_values),
        "coupling_phase_exponents",
    )
    return [
        GDPHullRelaxationStage(
            penalty=float(penalty),
            grayness_limit=float(limit),
            heaviside_beta=float(beta),
            solid_displacement_coupling_penalty=float(coupling),
            solid_displacement_coupling_phase_law=coupling_phase_law,
            solid_displacement_coupling_phase_exponent=float(exponent),
            solid_displacement_coupling_ramp_q=float(coupling_ramp_q),
            solid_displacement_coupling_norm=coupling_norm,
            solid_displacement_coupling_mode=coupling_mode,
        )
        for beta, limit, coupling, exponent in zip(beta_values, limit_values, coupling_values, exponent_values)
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run projected GDP-Hull NLP relaxation with weak grayness constraints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", default="cantilever")
    parser.add_argument("--nelx", type=_positive_int, default=60)
    parser.add_argument("--nely", type=_positive_int, default=20)
    parser.add_argument("--length", type=_positive_float, default=3.0)
    parser.add_argument("--height", type=_positive_float, default=1.0)
    parser.add_argument("--volfrac", type=_bounded_fraction, default=0.4)
    parser.add_argument("--u-bound", type=_positive_float, default=10000.0)
    parser.add_argument("--ipopt-max-iter", type=_positive_int, default=1200)
    parser.add_argument("--ipopt-tol", type=_positive_float, default=1e-7)
    parser.add_argument("--phase-filter-radius", type=_positive_float, default=1.5)
    parser.add_argument("--heaviside-eta", type=_open_unit_float, default=0.5)
    parser.add_argument("--betas", default="1,2,4,8,16,32")
    parser.add_argument("--grayness-limits", default="0.96,0.958,0.956,0.954,0.952,0.95")
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument("--coupling-penalties", default="0")
    parser.add_argument("--coupling-phase-law", choices=["linear", "simp", "ramp"], default="linear")
    parser.add_argument("--coupling-phase-exponent", type=_positive_float, default=1.0)
    parser.add_argument("--coupling-phase-exponents", default=None)
    parser.add_argument("--coupling-ramp-q", type=float, default=8.0)
    parser.add_argument("--coupling-norm", choices=["bound", "energy"], default="bound")
    parser.add_argument("--coupling-mode", choices=["penalty", "equality"], default="penalty")
    parser.add_argument("--initial-solution", default=None)
    parser.add_argument("--initial-density-field", choices=["ys", "raw_ys", "filtered_ys"], default="raw_ys")
    parser.add_argument("--connectivity", dest="connectivity", action="store_true", default=False)
    parser.add_argument("--no-connectivity", dest="connectivity", action="store_false")
    parser.add_argument("--fix-load-elements", dest="fix_load_elements", action="store_true", default=True)
    parser.add_argument("--no-fix-load-elements", dest="fix_load_elements", action="store_false")
    parser.add_argument("--tight-displacement-bounds", dest="tight_displacement_bounds", action="store_true", default=False)
    parser.add_argument("--no-tight-displacement-bounds", dest="tight_displacement_bounds", action="store_false")
    parser.add_argument("--tight-bound-safety-factor", type=_positive_float, default=20.0)
    parser.add_argument("--tight-bound-floor-fraction", type=_positive_float, default=1e-3)
    parser.add_argument(
        "--full-solid-compliance-lower-bound",
        dest="full_solid_compliance_lower_bound",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-full-solid-compliance-lower-bound",
        dest="full_solid_compliance_lower_bound",
        action="store_false",
    )
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args(argv)


def write_projected_grayness_metrics_csv(solution, output_path: Path) -> None:
    fields = [
        "stage_index",
        "heaviside_beta",
        "grayness_limit",
        "grayness_slack",
        "continuous_compliance",
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in solution.stage_results:
            limit = None if stage.grayness_limit is None else float(stage.grayness_limit)
            writer.writerow(
                {
                    "stage_index": stage.stage_index,
                    "heaviside_beta": stage.heaviside_beta,
                    "grayness_limit": limit,
                    "grayness_slack": "" if limit is None else limit - float(stage.global_grayness),
                    "continuous_compliance": stage.compliance,
                    "global_grayness": stage.global_grayness,
                    "max_element_grayness": stage.max_element_grayness,
                    "volume_fraction": stage.volume_fraction,
                    "solid_displacement_coupling_penalty": stage.solid_displacement_coupling_penalty,
                    "solid_displacement_coupling": stage.solid_displacement_coupling,
                    "solid_displacement_coupling_phase_law": stage.solid_displacement_coupling_phase_law,
                    "solid_displacement_coupling_phase_exponent": stage.solid_displacement_coupling_phase_exponent,
                    "solid_displacement_coupling_ramp_q": stage.solid_displacement_coupling_ramp_q,
                    "solid_displacement_coupling_norm": stage.solid_displacement_coupling_norm,
                    "solid_displacement_coupling_mode": stage.solid_displacement_coupling_mode,
                    "solver_status": stage.solver_status,
                    "termination_condition": stage.termination_condition,
                    "elapsed_seconds": stage.elapsed_seconds,
                }
            )


def _save_beta_history_plot(solution, output_path: Path, attribute: str, ylabel: str, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = [float(stage.heaviside_beta) for stage in solution.stage_results]
    y = [float(getattr(stage, attribute)) for stage in solution.stage_results]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    try:
        ax.plot(x, y, marker="o", linewidth=1.8, label=ylabel)
        if attribute == "global_grayness":
            limits = [
                float(stage.grayness_limit)
                for stage in solution.stage_results
                if stage.grayness_limit is not None
            ]
            if len(limits) == len(x):
                ax.plot(x, limits, marker="s", linewidth=1.4, linestyle="--", label="epsilon")
                ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Heaviside beta")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
    finally:
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from tpo.gdp_hull_relaxation import (
        GDPHullRelaxationConfig,
        solve_gdp_hull_relaxation_continuation,
    )
    from tpo.gdp_hull_relaxation_io import (
        build_gdp_hull_relaxation_summary,
        load_gdp_hull_relaxation_initial_solution_npz,
        write_gdp_hull_relaxation_metrics_summary_csv,
        write_gdp_hull_relaxation_solution_npz,
        write_gdp_hull_relaxation_stage_history_csv,
        write_gdp_hull_relaxation_summary_json,
    )
    from tpo.gdp_hull_relaxation_diagnostics import (
        write_gdp_hull_relaxation_diagnostics_csv,
        write_gdp_hull_relaxation_diagnostics_json,
    )
    from tpo.plotting import save_density_plot, save_displacement_plot
    from tpo.runtime import RESULTS_ROOT

    config = GDPHullRelaxationConfig(
        benchmark=canonical_benchmark_label(args.benchmark),
        nelx=args.nelx,
        nely=args.nely,
        length=args.length,
        height=args.height,
        volfrac=args.volfrac,
        displacement_bound=args.u_bound,
        ipopt_max_iter=args.ipopt_max_iter,
        ipopt_tol=args.ipopt_tol,
        tee=args.tee,
        use_element_connectivity=args.connectivity,
        fix_load_elements=args.fix_load_elements,
        phase_filter_radius=args.phase_filter_radius,
        heaviside_eta=args.heaviside_eta,
        use_tight_displacement_bounds=args.tight_displacement_bounds,
        tight_displacement_bound_safety_factor=args.tight_bound_safety_factor,
        tight_displacement_bound_floor_fraction=args.tight_bound_floor_fraction,
        enforce_full_solid_compliance_lower_bound=args.full_solid_compliance_lower_bound,
    )
    stages = parse_projected_grayness_stages(
        args.betas,
        args.grayness_limits,
        penalty=args.penalty,
        coupling_penalties=args.coupling_penalties,
        coupling_phase_law=args.coupling_phase_law,
        coupling_phase_exponent=args.coupling_phase_exponent,
        coupling_phase_exponents=args.coupling_phase_exponents,
        coupling_ramp_q=args.coupling_ramp_q,
        coupling_norm=args.coupling_norm,
        coupling_mode=args.coupling_mode,
    )
    initial_solution = (
        None
        if args.initial_solution is None
        else load_gdp_hull_relaxation_initial_solution_npz(
            config,
            Path(args.initial_solution),
            density_field=args.initial_density_field,
        )
    )
    solution = solve_gdp_hull_relaxation_continuation(config, stages, initial=initial_solution)
    out_dir = RESULTS_ROOT / result_directory_name(args.benchmark, args.nelx, args.nely)
    out_dir.mkdir(parents=True, exist_ok=True)

    if solution.raw_ys is not None:
        save_density_plot(
            solution.model_data.mesh,
            solution.raw_ys,
            out_dir / "raw_ys_density.png",
            title="Raw design solid fraction",
        )
    if solution.filtered_ys is not None:
        save_density_plot(
            solution.model_data.mesh,
            solution.filtered_ys,
            out_dir / "filtered_ys_density.png",
            title="Filtered solid fraction",
        )
    save_density_plot(
        solution.model_data.mesh,
        solution.ys,
        out_dir / "projected_ys_density.png",
        title="Projected weak-grayness solid fraction",
    )
    save_density_plot(
        solution.model_data.mesh,
        4.0 * solution.ys * solution.yv,
        out_dir / "projected_grayness.png",
        title="Projected weak-grayness field",
    )
    save_density_plot(
        solution.model_data.mesh,
        solution.rounded_binary_density,
        out_dir / "rounded_density.png",
        title="Rounded binary topology from projected density",
    )
    if solution.rounded_binary_displacement is not None:
        save_displacement_plot(
            solution.model_data.mesh,
            solution.rounded_binary_displacement,
            out_dir / "rounded_displacement.png",
        )

    write_gdp_hull_relaxation_summary_json(solution, out_dir / "summary.json")
    write_gdp_hull_relaxation_metrics_summary_csv(solution, out_dir / "metrics_summary.csv")
    write_gdp_hull_relaxation_stage_history_csv(solution, out_dir / "stage_history.csv")
    write_gdp_hull_relaxation_solution_npz(solution, out_dir / "solution.npz")
    write_gdp_hull_relaxation_diagnostics_json(solution, out_dir / "diagnostics.json")
    write_gdp_hull_relaxation_diagnostics_csv(solution, out_dir / "diagnostics.csv")
    write_projected_grayness_metrics_csv(solution, out_dir / "projected_grayness_metrics.csv")
    _save_beta_history_plot(
        solution,
        out_dir / "beta_grayness_curve.png",
        "global_grayness",
        "P(bar y)",
        "Projected GDP-Hull weak grayness constraint",
    )
    _save_beta_history_plot(
        solution,
        out_dir / "beta_compliance_curve.png",
        "compliance",
        "Compliance",
        "Continuous compliance by Heaviside beta",
    )

    summary = build_gdp_hull_relaxation_summary(solution)
    print(f"Result directory: {out_dir}")
    print(f"Continuous compliance: {summary['continuous_compliance']:.8g}")
    print(f"Global grayness: {summary['global_grayness']:.8g}")
    print(f"Rounded compliance: {summary['rounded_binary_compliance']}")
    print(f"Rounded response: {summary['rounded_response_status']}")
    if not summary["completed_all_stages"]:
        print(
            "Continuation warning: "
            f"failed at stage {summary['failed_stage_index']}: {summary['failed_stage_reason']}"
        )
    print(f"Elapsed seconds: {summary['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()
