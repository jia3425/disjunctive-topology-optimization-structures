"""Record-only driver for the existing uniform-start structural GDP route."""
from pathlib import Path
from dataclasses import asdict, replace
import hashlib
import json
import sys
import time
import traceback
import numpy as np

SOURCE = Path(r'D:\Case\TPO\src')
sys.path.insert(0, str(SOURCE))
import pyomo.environ as pyo
from tpo.gdp_hull_relaxation import (
    GDPHullRelaxationConfig, GDPHullRelaxationStage,
    build_gdp_hull_relaxation_model, _extract_stage_solution,
)
from tpo.gdp_hull_relaxation_io import (
    write_gdp_hull_relaxation_solution_npz,
    write_gdp_hull_relaxation_summary_json,
    write_gdp_hull_relaxation_stage_history_csv,
)
from tpo.runtime import SolverSettings, create_ipopt_solver

ROOT = Path(__file__).resolve().parent


def make_config(nelx=60, nely=20):
    return GDPHullRelaxationConfig(
        benchmark='cantilever', nelx=nelx, nely=nely, length=3, height=1,
        volfrac=.4, phase_filter_radius=1.5, heaviside_eta=.5,
        fix_load_elements=True, use_tight_displacement_bounds=True,
        tight_displacement_bound_safety_factor=50,
        enforce_full_solid_compliance_lower_bound=True,
        ipopt_max_iter=1200, ipopt_tol=1e-7, tee=True,
    )


def make_stages():
    schedule = [(1, b, g) for b, g in zip([1,1.5,2,2.5,3], [.96,.80,.55,.35,.22])]
    schedule += [(2, b, g) for b, g in zip([3,4,5,6,8,10,12,14,16], [.22,.15,.11,.08,.04,.03,.02,.015,.01])]
    return [GDPHullRelaxationStage(
        penalty=0, grayness_limit=g, heaviside_beta=b,
        solid_displacement_coupling_phase_law='simp',
        solid_displacement_coupling_phase_exponent=p,
        solid_displacement_coupling_norm='bound',
        solid_displacement_coupling_mode='equality',
    ) for p, b, g in schedule]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def capture_model(model, data, directory, metadata):
    """Keep exact (unclipped) solver values; initial and nonoptimal states are valid records."""
    directory.mkdir(parents=True, exist_ok=True)
    variables = list(model.component_data_objects(pyo.Var))
    values = np.array([pyo.value(v) for v in variables], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('Cannot plot a state containing nonfinite variable values')
    field = lambda component: np.array([pyo.value(component[e]) for e in model.E], dtype=float)
    ys = field(model.solid_phase)
    displacement = np.zeros(data.mesh.total_dofs)
    displacement[data.mesh.free_dofs] = [pyo.value(model.u[f]) for f in model.F]
    np.savez_compressed(
        directory/'state.npz', ys=ys, raw_ys=field(model.ys),
        filtered_ys=field(model.ys_tilde), raw_ys_fixed=np.array([model.ys[e].fixed for e in model.E]),
        displacement=displacement,
        solid_displacement_components=np.array([[pyo.value(model.us[e,a]) for a in model.A] for e in model.E]),
        nodes=data.mesh.nodes, elements=data.mesh.elements,
        element_dofs=data.mesh.element_dofs, force=data.mesh.force,
        fixed_dofs=data.mesh.fixed_dofs, free_dofs=data.mesh.free_dofs,
        variable_names=np.array([v.name for v in variables]), variable_values=values,
    )
    residuals = {}
    for c in model.component_data_objects(pyo.Constraint, active=True):
        body = float(pyo.value(c.body))
        violation = max(0., float(pyo.value(c.lower))-body if c.has_lb() else 0., body-float(pyo.value(c.upper)) if c.has_ub() else 0.)
        if not np.isfinite(body):
            raise ValueError('Nonfinite constraint: '+c.name)
        parent = c.parent_component().name
        residuals[parent] = max(residuals.get(parent, 0.), violation)
    bounds = [max(0., float(pyo.value(v.lb))-x if v.has_lb() else 0., x-float(pyo.value(v.ub)) if v.has_ub() else 0.) for v,x in zip(variables, values)]
    report = dict(metadata, volume_fraction=float(ys.mean()), global_grayness=float(np.mean(4*ys*(1-ys))),
        continuous_compliance=float(data.mesh.force@displacement), ys_min=float(ys.min()), ys_max=float(ys.max()),
        max_constraint_violation=max(residuals.values(), default=0.), max_bound_violation=max(bounds, default=0.),
        constraint_group_max=residuals, field_sha256=hashlib.sha256((directory/'state.npz').read_bytes()).hexdigest())
    write_json(directory/'state.json', report)
    return report


def main():
    config, stages = make_config(), make_stages()
    if (ROOT/'stage_00_initial').exists():
        raise RuntimeError('This run already has snapshots; do not overwrite it')
    source_files = [SOURCE/'tpo'/name for name in ['gdp_hull_relaxation.py','gdp_hull_relaxation_io.py','runtime.py']]
    provenance = {
        'config': asdict(config), 'stages': [asdict(s) for s in stages],
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'initialization': 'Default uniform raw free phase 0.4; fixed load elements 1; zero displacement. Initial state is not an equilibrium solution.',
        'p_transition': 'At stage 6 only, reset local us to ys**2 times element global displacement per existing documented route.',
        'continuous_fields_only': True,
    }
    write_json(ROOT/'run_config.json', provenance)
    solution, history, records = None, [], []
    previous_p = None
    for index, stage in enumerate(stages, start=1):
        p = stage.solid_displacement_coupling_phase_exponent
        transition = solution is not None and p != previous_p
        if transition:
            solution = replace(solution, solid_displacement_components=solution.ys[:,None]**p * solution.full_displacement[solution.model_data.mesh.element_dofs])
        model, data = build_gdp_hull_relaxation_model(config, stage, initial=solution)
        directory = ROOT/f'stage_{index:02d}'
        directory.mkdir()
        meta = {'stage_index': index, 'stage': asdict(stage), 'previous_stage': index-1,
                'initial_source': 'uniform' if index == 1 else str(ROOT/f'stage_{index-1:02d}'/'solution.npz'),
                'local_displacement_reinitialized': transition}
        if index == 1:
            capture_model(model, data, ROOT/'stage_00_initial', dict(meta, stage_index=0, kind='initial', termination_condition='not_run'))
        capture_model(model, data, directory/'start', dict(meta, kind='stage_start', termination_condition='not_run'))
        write_json(ROOT/'progress.json', {'status':'running','active_stage':index,'total_stages':len(stages),'finished':records})
        print(f'BEGIN_STAGE {index} p={p} beta={stage.heaviside_beta} gray_cap={stage.grayness_limit}', flush=True)
        solver = create_ipopt_solver(SolverSettings(tol=config.ipopt_tol, max_iter=config.ipopt_max_iter, tee=True))
        started = time.perf_counter()
        results = solver.solve(model, tee=True, load_solutions=False, logfile=str(directory/'solver.log'))
        elapsed = time.perf_counter()-started
        if len(results.solution) == 0:
            raise RuntimeError(f'No readable solution returned at stage {index}: {results.solver.termination_condition}')
        model.solutions.load_from(results)
        report = capture_model(model, data, directory, dict(meta, kind='stage_end', solver_status=str(results.solver.status),
            termination_condition=str(results.solver.termination_condition), elapsed_seconds=elapsed))
        solution = _extract_stage_solution(model, data, results, stage, index, elapsed, history)
        history = list(solution.stage_results)
        write_gdp_hull_relaxation_solution_npz(solution, directory/'solution.npz')
        write_gdp_hull_relaxation_summary_json(solution, directory/'summary.json')
        write_gdp_hull_relaxation_stage_history_csv(solution, ROOT/'stage_history.csv')
        records.append({k:report[k] for k in ['stage_index','termination_condition','global_grayness','volume_fraction','continuous_compliance','max_constraint_violation','elapsed_seconds']})
        previous_p = p
        print('END_STAGE '+json.dumps(records[-1]), flush=True)
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p,digest in provenance['source_hashes'].items())
    write_json(ROOT/'progress.json', {'status':'finished','finished':records,'source_hashes_unchanged':unchanged})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        progress_path = ROOT/'progress.json'
        progress = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
        progress.update(status='stopped_with_error', error=str(exc))
        write_json(progress_path, progress)
        raise
