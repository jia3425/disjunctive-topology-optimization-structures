"""Fresh uniform-start structural runs; legacy MBB or refined L-bracket continuation."""
from pathlib import Path
from dataclasses import asdict, replace
from datetime import datetime
import argparse
import hashlib
import importlib.util
import json
import os
import time
import traceback

# Keep the small sparse solves single-threaded, as in the recent SIMP run.
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

HELPER_PATH = Path('C:/Users/JQ/Documents/拓扑优化/paper/figures/figure_1/panel_c/solid_run_20260909/run_solid.py')
spec = importlib.util.spec_from_file_location('existing_solid_recorder', HELPER_PATH)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def make_config(nelx=None, nely=None, benchmark='mbb'):
    nelx = 60 if nelx is None else nelx
    nely = (60 if benchmark == 'lbracket' else 20) if nely is None else nely
    config = replace(helper.make_config(nelx, nely), benchmark=benchmark)
    return replace(config, length=2., height=2.) if benchmark == 'lbracket' else config


def make_stages(benchmark='mbb'):
    old = helper.make_stages()
    if benchmark != 'lbracket':
        return old
    # Keep the uniform p1 path; soften p1->p2 and use unit beta steps thereafter.
    stages = old[:5]+[replace(old[5], solid_displacement_coupling_phase_exponent=1.5)]
    caps = [.22,.15,.11,.08,.06,.04,.035,.03,.025,.02,.0175,.015,.0125,.01]
    stages += [replace(old[5], heaviside_beta=float(b), grayness_limit=g) for b,g in zip(range(3,17),caps)]
    return stages


def equilibrate_initial_state(model, data, stage):
    """Set displacement only for the uniform design; no density optimization or regularization."""
    from tpo.fea import solve_linear_response
    np = helper.np
    factors = np.array([helper.pyo.value(model.gdp_solid_phase_scale[e]) for e in model.E])
    free_u, full_u = solve_linear_response(data.mesh, data.element_ke, factors)
    if np.any(np.abs(full_u)>data.displacement_bounds*(1+1e-12)):
        raise RuntimeError('Uniform equilibrium exceeds existing displacement bounds')
    for i,value in enumerate(free_u):
        model.u[i].set_value(float(value))
    local = factors[:,None]*full_u[data.mesh.element_dofs]
    for e in model.E:
        for a in model.A:
            model.us[e,a].set_value(float(local[e,a]))


def is_accepted(report):
    return (report['termination_condition'] in ['optimal', 'locallyOptimal']
            and report['max_constraint_violation'] <= 1e-6
            and report['max_bound_violation'] <= 1e-6)


def reserve_output(path):
    path.mkdir(parents=True, exist_ok=False)


def resume_provenance(directory):
    report = json.loads((directory/'state.json').read_text(encoding='utf-8'))
    if report.get('accepted') is not True or not is_accepted(report):
        raise ValueError('Resume only from a converged and accepted stage')
    previous = json.loads((directory.parent/'run_config.json').read_text(encoding='utf-8'))
    origin = Path(previous.get('uniform_origin',directory.parent))
    origin_config = json.loads((origin/'run_config.json').read_text(encoding='utf-8'))
    if origin_config['initial_solution_file'] is not None:
        raise ValueError('Cannot verify a fresh uniform origin')
    with helper.np.load(origin/'stage_00_initial/state.npz') as initial:
        fixed=initial['raw_ys_fixed']; raw=initial['raw_ys']
        if not (helper.np.all(raw[~fixed]==origin_config['config']['volfrac']) and helper.np.all(raw[fixed]==1.)):
            raise ValueError('Origin raw design is not uniform')
    return {'uniform_origin':str(origin),'source':str(directory/'solution.npz'),
            'previous_p':report['stage']['solid_displacement_coupling_phase_exponent'],
            'source_config':previous['config']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--benchmark',choices=['mbb','lbracket'],default='mbb')
    parser.add_argument('--max-cpu-time',type=float,default=600.)
    parser.add_argument('--resume-from',type=Path,help='Accepted stage directory in this fresh-uniform lineage')
    parser.add_argument('--schedule',type=Path,help='JSON list of [p,beta,grayness_cap] triples')
    args = parser.parse_args()
    if not helper.np.isfinite(args.max_cpu_time) or args.max_cpu_time<=0:
        raise ValueError('max-cpu-time must be finite and positive')
    root = args.output or Path('D:/Case/TPO/results') / (f'gdp_{args.benchmark}_uniform_rerun_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    reserve_output(root)
    config, stages = make_config(benchmark=args.benchmark), make_stages(args.benchmark)
    if args.schedule:
        triples=json.loads(args.schedule.read_text(encoding='utf-8'))
        if not triples:
            raise ValueError('Schedule must not be empty')
        stages=[replace(stages[-1],solid_displacement_coupling_phase_exponent=float(p),
                        heaviside_beta=float(b),grayness_limit=float(g)) for p,b,g in triples]
    resume = resume_provenance(args.resume_from) if args.resume_from else None
    if resume and resume['source_config'] != asdict(config):
        raise ValueError('Resume configuration differs from the current physical model')
    watched = [HELPER_PATH] + [helper.SOURCE/'tpo'/name for name in ['gdp_hull_relaxation.py', 'gdp_hull_relaxation_io.py', 'runtime.py']]
    base = Path('D:/Case/TPO/results')
    tag = f'{args.benchmark}_{config.nelx}x{config.nely}'
    for label in [f'gdp_hull_relaxation_projected_weak_grayness_{tag}_uniform_p1_seed',
                  f'gdp_hull_relaxation_projected_weak_grayness_{tag}_uniform_p1_to_p2']:
        watched.extend([base/label/'solution.npz', base/label/'summary.json'])
    figure_root = Path('C:/Users/JQ/Documents/拓扑优化/paper/figures/figure_2')
    for panel,stem in [('panel_a','figure_2a_cantilever'),('panel_b','figure_2b_mbb'),('panel_c','figure_2c_lbracket')]:
        watched.extend([figure_root/panel/(stem+ext) for ext in ['.svg','.png']])
    digests = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    if resume:
        for path in [Path(resume['source']),args.resume_from/'state.json',Path(resume['uniform_origin'])/'stage_00_initial/state.npz']:
            digests[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    helper.write_json(root/'run_config.json', {
        'config':asdict(config), 'stages':[asdict(s) for s in stages],
        'initialization':('Resume from an accepted stage of the same freshly uniform-started run; no SIMP/OC or historical design.' if resume else 'No old solution loaded. All free raw design values 0.4, fixed load elements 1. '+('Initial displacement from equilibrium of the initialized phase field.' if args.benchmark=='lbracket' else 'Initial displacement zero.')),
        'initial_solution_file':resume['source'] if resume else None,
        'uniform_origin':resume['uniform_origin'] if resume else str(root),
        'initial_state_is_equilibrium':args.benchmark=='lbracket' and not resume,
        'p_transition':'When p changes, initialize us to ys**p times element displacement; raw design is unchanged.',
        'solver_extra_options':{'max_cpu_time':args.max_cpu_time, 'threads':1},
        'tracked_element':117 if args.benchmark=='lbracket' else None,
        'protected_hashes':digests, 'driver':str(Path(__file__).resolve()),
    })
    print('RESULT_DIRECTORY '+str(root), flush=True)
    records, solution, history = [], None, []
    if resume:
        from tpo.gdp_hull_relaxation_io import load_gdp_hull_relaxation_initial_solution_npz
        solution=load_gdp_hull_relaxation_initial_solution_npz(config,Path(resume['source']),density_field='ys')
    started_run = time.perf_counter()
    try:
        for index, stage in enumerate(stages, start=1):
            p = stage.solid_displacement_coupling_phase_exponent
            previous_p = stages[index-2].solid_displacement_coupling_phase_exponent if index>1 else (resume['previous_p'] if resume else p)
            transition = solution is not None and p != previous_p
            if transition:
                solution = replace(solution, solid_displacement_components=
                    solution.ys[:,None]**p * solution.full_displacement[solution.model_data.mesh.element_dofs])
            model, data = helper.build_gdp_hull_relaxation_model(config, stage, initial=solution)
            if index==1 and args.benchmark=='lbracket' and not resume:
                equilibrate_initial_state(model,data,stage)
            if index==1 and resume:
                with helper.np.load(resume['source']) as archive:
                    if not all(helper.np.array_equal(archive[k],getattr(data.mesh,k)) for k in ['nodes','elements','force','fixed_dofs']):
                        raise ValueError('Resume geometry/boundary mismatch')
            directory = root/f'stage_{index:02d}'
            directory.mkdir()
            meta = {'stage_index':index, 'stage':asdict(stage),
                    'initial_source':('uniform' if not resume else resume['source']) if index==1 else str(root/f'stage_{index-1:02d}'/'solution.npz'),
                    'local_displacement_reinitialized':transition}
            if index == 1 and not resume:
                helper.capture_model(model, data, root/'stage_00_initial',
                    dict(meta, stage_index=0, kind='initial', termination_condition='not_run'))
            helper.write_json(root/'progress.json', {'status':'running','active_stage':index,'total_stages':len(stages),'finished':records})
            print(f'BEGIN_STAGE {index}/{len(stages)} p={p} beta={stage.heaviside_beta} gray_cap={stage.grayness_limit}', flush=True)
            solver = helper.create_ipopt_solver(helper.SolverSettings(tol=config.ipopt_tol,
                max_iter=config.ipopt_max_iter, tee=False, output_file=str(directory/'ipopt_live.log')))
            solver.options['max_cpu_time'] = args.max_cpu_time
            started = time.perf_counter()
            results = solver.solve(model, tee=False, load_solutions=False, logfile=str(directory/'solver.log'))
            elapsed = time.perf_counter()-started
            if len(results.solution) == 0:
                raise RuntimeError(f'No readable solution at stage {index}: {results.solver.termination_condition}')
            model.solutions.load_from(results)
            report = helper.capture_model(model, data, directory, dict(meta, kind='stage_end',
                solver_status=str(results.solver.status), termination_condition=str(results.solver.termination_condition), elapsed_seconds=elapsed))
            report['accepted'] = is_accepted(report)
            if args.benchmark=='lbracket':
                report['tracked_element_solid_fraction'] = float(helper.pyo.value(model.solid_phase[117]))
                report['tracked_element_raw_fraction'] = float(helper.pyo.value(model.ys[117]))
            helper.write_json(directory/'state.json', report)
            records.append({k:report[k] for k in ['stage_index','termination_condition','accepted','global_grayness',
                'volume_fraction','continuous_compliance','max_constraint_violation','elapsed_seconds']})
            if args.benchmark=='lbracket':
                records[-1]['tracked_element_solid_fraction'] = report['tracked_element_solid_fraction']
            print('END_STAGE '+json.dumps(records[-1]), flush=True)
            if not report['accepted']:
                raise RuntimeError(f'Stage {index} did not meet convergence and residual requirements; state retained, no further stage run.')
            solution = helper._extract_stage_solution(model, data, results, stage, index, elapsed, history)
            history = list(solution.stage_results)
            helper.write_gdp_hull_relaxation_solution_npz(solution, directory/'solution.npz')
            helper.write_gdp_hull_relaxation_summary_json(solution, directory/'summary.json')
            helper.write_gdp_hull_relaxation_stage_history_csv(solution, root/'stage_history.csv')
        helper.write_gdp_hull_relaxation_solution_npz(solution, root/'solution.npz')
        helper.write_gdp_hull_relaxation_summary_json(solution, root/'summary.json')
        helper.write_json(root/'progress.json', {'status':'completed','finished':records,
            'elapsed_seconds':time.perf_counter()-started_run,
            'protected_files_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in digests.items())})
        print('COMPLETED '+str(root), flush=True)
    except Exception as exc:
        helper.write_json(root/'progress.json', {'status':'stopped_incomplete','finished':records,
            'error':str(exc),'elapsed_seconds':time.perf_counter()-started_run,
            'protected_files_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in digests.items())})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
