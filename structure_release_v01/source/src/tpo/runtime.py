from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyomo.environ as pyo


PROJECT_ROOT = Path(r"D:\Case\TPO")
SRC_ROOT = PROJECT_ROOT / "src"
RESULTS_ROOT = PROJECT_ROOT / "results"
IPOPT_EXECUTABLE = Path(r"C:\Users\JQ\AppData\Local\idaes\bin\ipopt.exe")


@dataclass(frozen=True)
class SolverSettings:
    name: str = "ipopt"
    tol: float = 1e-7
    max_iter: int = 1200
    linear_solver: str = "mumps"
    tee: bool = False
    acceptable_tol: float | None = None
    acceptable_iter: int | None = None
    acceptable_constr_viol_tol: float | None = None
    acceptable_dual_inf_tol: float | None = None
    acceptable_compl_inf_tol: float | None = None
    output_file: str | None = None
    file_print_level: int | None = None
    bound_push: float | None = None
    bound_frac: float | None = None


def create_ipopt_solver(settings: SolverSettings | None = None) -> Any:
    settings = settings or SolverSettings()
    solver = pyo.SolverFactory(settings.name, executable=str(IPOPT_EXECUTABLE))
    solver.options["tol"] = settings.tol
    solver.options["max_iter"] = settings.max_iter
    solver.options["linear_solver"] = settings.linear_solver
    optional_options = {
        "acceptable_tol": settings.acceptable_tol,
        "OF_acceptable_iter": settings.acceptable_iter,
        "acceptable_constr_viol_tol": settings.acceptable_constr_viol_tol,
        "acceptable_dual_inf_tol": settings.acceptable_dual_inf_tol,
        "acceptable_compl_inf_tol": settings.acceptable_compl_inf_tol,
        "OF_output_file": settings.output_file,
        "OF_file_print_level": settings.file_print_level,
        "bound_push": settings.bound_push,
        "bound_frac": settings.bound_frac,
    }
    for option, value in optional_options.items():
        if value is not None:
            solver.options[option] = value
    return solver


def assert_runtime_available() -> None:
    if not IPOPT_EXECUTABLE.exists():
        raise FileNotFoundError(f"IPOPT executable not found: {IPOPT_EXECUTABLE}")
    solver = create_ipopt_solver()
    if not solver.available(False):
        raise RuntimeError(f"Pyomo cannot use IPOPT executable: {IPOPT_EXECUTABLE}")
