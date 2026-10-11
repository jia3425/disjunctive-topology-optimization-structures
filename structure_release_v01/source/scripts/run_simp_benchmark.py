from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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


def benchmark_defaults(benchmark: str) -> tuple[int, int, float, float]:
    label = canonical_benchmark_label(benchmark)
    if label == "cantilever":
        return (30, 10, 3.0, 1.0)
    if label == "mbb":
        return (60, 20, 3.0, 1.0)
    if label == "lbracket":
        return (60, 60, 2.0, 2.0)
    raise ValueError("unknown benchmark; expected one of cantilever, mbb, or lbracket")


def result_directory_name(benchmark: str, nelx: int, nely: int) -> str:
    label = canonical_benchmark_label(benchmark)
    if label not in {"cantilever", "mbb", "lbracket"}:
        raise ValueError("unknown benchmark; expected one of cantilever, mbb, or lbracket")
    return f"simp_{label}_{int(nelx)}x{int(nely)}"


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _positive_float(value: str) -> float:
    result = _finite_float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _bounded_fraction(value: str) -> float:
    result = _finite_float(value)
    if not (0.0 < result <= 1.0):
        raise argparse.ArgumentTypeError("must satisfy 0 < value <= 1")
    return result


def _penalties(value: str) -> tuple[float, ...]:
    try:
        penalties = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("penalties must be comma-separated numbers") from exc
    if not penalties:
        raise argparse.ArgumentTypeError("penalties must contain at least one value")
    if any((not math.isfinite(penalty)) or penalty <= 0.0 for penalty in penalties):
        raise argparse.ArgumentTypeError("penalties must be finite positive values")
    return penalties


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a SIMP topology optimization benchmark with Pyomo and IPOPT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        default="cantilever",
        help="Benchmark: cantilever, mbb, or lbracket.",
    )
    parser.add_argument("--nelx", type=_positive_int, default=None, help="Elements along length.")
    parser.add_argument("--nely", type=_positive_int, default=None, help="Elements along height.")
    parser.add_argument("--length", type=_positive_float, default=None, help="Physical length.")
    parser.add_argument("--height", type=_positive_float, default=None, help="Physical height.")
    parser.add_argument("--volfrac", type=_bounded_fraction, default=0.4, help="Volume fraction.")
    parser.add_argument(
        "--rmin",
        type=_positive_float,
        default=1.5,
        help="Density filter radius in element widths.",
    )
    parser.add_argument(
        "--penalties",
        type=_penalties,
        default=(1.0, 2.0, 3.0),
        help="Comma-separated SIMP continuation penalties.",
    )
    parser.add_argument("--tol", type=_positive_float, default=1e-7, help="IPOPT tolerance.")
    parser.add_argument("--max-iter", type=_positive_int, default=1200, help="IPOPT max iterations per stage.")
    parser.add_argument("--linear-solver", default="mumps", help="IPOPT linear solver.")
    parser.add_argument("--tee", action="store_true", help="Stream IPOPT output.")

    args = parser.parse_args(argv)
    args.benchmark = canonical_benchmark_label(args.benchmark)
    default_nelx, default_nely, default_length, default_height = benchmark_defaults(args.benchmark)
    if args.nelx is None:
        args.nelx = default_nelx
    if args.nely is None:
        args.nely = default_nely
    if args.length is None:
        args.length = default_length
    if args.height is None:
        args.height = default_height
    return args


def _penalty_file_label(penalty: float) -> str:
    return f"p{penalty:g}".replace("-", "m").replace(".", "p")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from tpo.io import (
        build_summary,
        write_convergence_csv,
        write_metrics_summary_csv,
        write_solution_npz,
        write_summary_json,
    )
    from tpo.plotting import (
        save_density_evolution_plot,
        save_density_plot,
        save_displacement_plot,
        save_objective_history_plot,
        save_time_history_plot,
        save_volume_history_plot,
    )
    from tpo.runtime import RESULTS_ROOT, SolverSettings, assert_runtime_available
    from tpo.simp_model import SimpConfig, solve_simp_continuation

    assert_runtime_available()

    config = SimpConfig(
        benchmark=args.benchmark,
        nelx=args.nelx,
        nely=args.nely,
        length=args.length,
        height=args.height,
        volfrac=args.volfrac,
        rmin=args.rmin,
        penalties=args.penalties,
    )
    solution = solve_simp_continuation(
        config=config,
        solver_settings=SolverSettings(
            tol=args.tol,
            max_iter=args.max_iter,
            linear_solver=args.linear_solver,
            tee=args.tee,
        ),
        tee=args.tee,
    )

    out_dir = RESULTS_ROOT / result_directory_name(args.benchmark, args.nelx, args.nely)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_density_plot(solution.model_data.mesh, solution.physical_density, out_dir / "density.png")
    save_displacement_plot(
        solution.model_data.mesh,
        solution.full_displacement,
        out_dir / "displacement.png",
    )
    write_convergence_csv(solution, out_dir / "convergence.csv")
    write_metrics_summary_csv(solution, out_dir / "metrics_summary.csv")
    write_summary_json(solution, out_dir / "summary.json")
    write_solution_npz(solution, out_dir / "solution.npz")
    for snapshot in solution.stage_snapshots:
        save_density_plot(
            solution.model_data.mesh,
            snapshot.physical_density,
            out_dir / f"stage_density_{_penalty_file_label(snapshot.penalty)}.png",
        )
    save_density_evolution_plot(
        solution.model_data.mesh,
        solution.stage_snapshots,
        out_dir / "density_evolution.png",
    )
    save_objective_history_plot(solution.stage_results, out_dir / "objective_history.png")
    save_volume_history_plot(solution.stage_results, out_dir / "volume_history.png")
    save_time_history_plot(solution.stage_results, out_dir / "time_history.png")

    summary = build_summary(solution)
    print(f"Result directory: {out_dir}")
    print(f"Benchmark: {summary['benchmark']}")
    print(f"Compliance: {summary['compliance']:.8g}")
    print(f"Volume fraction: {summary['actual_volume_fraction']:.8g}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']:.3f}")
    print(f"Termination: {summary['termination_condition']}")


if __name__ == "__main__":
    main()
