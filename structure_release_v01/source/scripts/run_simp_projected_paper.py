"""Independent projected-SIMP experiment; original code/results remain unchanged."""
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import sys
import time

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import check_optimal_termination

PROJECT = Path('D:/Case/TPO')
sys.path.insert(0, str(PROJECT/'src'))
from tpo.simp_model import SimpConfig, build_simp_model
from tpo.filtering import apply_density_filter
from tpo.fea import solve_linear_response, assemble_free_stiffness
from tpo.runtime import SolverSettings, create_ipopt_solver


def project(density, beta, eta=.5, symbolic=False):
    if not np.isfinite(beta) or beta < 0 or not 0 < eta < 1:
        raise ValueError('Require beta >= 0 and 0 < eta < 1')
    if beta == 0:
        return density
    tanh = pyo.tanh if symbolic else np.tanh
    return (np.tanh(beta*eta)+tanh(beta*(density-eta))) / (
        np.tanh(beta*eta)+np.tanh(beta*(1-eta)))


def stiffness_factors(config, physical, penalty):
    return config.emin+(config.e0-config.emin)*physical**penalty


def build_projected_model(config, raw, beta, eta=.5, penalty=3.):
    model, data = build_simp_model(config, penalty=penalty, initial_density=raw)
    # Existing stiffness, volume and equilibrium reference these ExpressionData
    # objects. Replace their expressions so projection is INSIDE the NLP.
    for e in model.E:
        filtered_expression = model.rho[e].expr
        model.rho[e].set_value(project(filtered_expression, beta, eta, symbolic=True))
    physical = project(apply_density_filter(data.filter_data, raw), beta, eta)
    factors = stiffness_factors(config, physical, penalty)
    free_u, _ = solve_linear_response(data.mesh, data.element_ke, factors)
    for i, value in enumerate(free_u):
        model.u[i].set_value(float(value))
    return model, data


def create_output(path):
    # Deliberately refuse even an existing empty directory.
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def make_config(benchmark, penalty=3., rmin=1.5):
    geometry = dict(nelx=60, nely=60, length=2., height=2.) if benchmark == 'lbracket' else dict(
        nelx=60, nely=20, length=3., height=1.)
    return SimpConfig(benchmark=benchmark, **geometry, volfrac=.4, rmin=rmin, penalties=(penalty,))


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def plot_comparison(old, new, mesh, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.font_manager import fontManager
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    cmap = LinearSegmentedColormap.from_list('solid_phase', ['#f5f7fa', '#8bbde0', '#075fa8'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), facecolor='white')
    fig.subplots_adjust(left=.035, right=.98, bottom=.29, top=.80, wspace=.17)
    for ax, values, title in zip(axes, [old, new], ['原SIMP：密度滤波', '新SIMP：密度滤波＋投影']):
        collection = PolyCollection(mesh.nodes[mesh.elements], array=values, cmap=cmap,
                                    norm=Normalize(0, 1), edgecolors='face', linewidths=.1,
                                    antialiaseds=False)
        ax.add_collection(collection)
        ax.add_patch(Rectangle((0, 0), 3, 1, fill=False, edgecolor='#9badbf', linewidth=.6))
        ax.plot([0, 0], [0, 1], color='#13263b', lw=1.4)
        for y in np.linspace(.03, .98, 11):
            ax.plot([-.09, 0], [y-.05, y], color='#536477', lw=.6)
        ax.plot([3, 3.10], [.5, .5], color='#13263b', lw=.8)
        ax.add_patch(FancyArrowPatch((3.10, .5), (3.10, .23), arrowstyle='-|>',
                                     mutation_scale=10, color='#13263b', lw=.9))
        ax.set(xlim=(-.11, 3.18), ylim=(-.03, 1.03), aspect='equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, color='#13263b', pad=13)
        ax.text(.5, -.18, f'体积分数 {values.mean():.2%}   ·   整体灰度 {np.mean(4*values*(1-values)):.2%}',
                transform=ax.transAxes, ha='center', color='#536477', fontsize=11)
    cax = fig.add_axes([.36, .14, .28, .03])
    bar = fig.colorbar(collection, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    bar.set_label('固体份额', fontsize=10)
    bar.outline.set_linewidth(.4)
    fig.savefig(output/'comparison_original_vs_projected.png', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['cantilever', 'mbb', 'lbracket'], default='cantilever')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--seed', type=Path, default=None)
    parser.add_argument('--betas', default='1,2,4,8,16')
    parser.add_argument('--penalty', type=float, default=3.)
    parser.add_argument('--eta', type=float, default=.5)
    parser.add_argument('--rmin', type=float, default=1.5)
    parser.add_argument('--oc-warmup', type=int, default=0, help='OC initialization iterations; NLP acceptance unchanged')
    parser.add_argument('--oc-move', type=float, default=.05)
    parser.add_argument('--max-cpu-time', type=float, default=300.)
    args = parser.parse_args()
    if not np.isfinite(args.max_cpu_time) or args.max_cpu_time <= 0:
        raise ValueError('max-cpu-time must be positive and finite')
    if not np.isfinite(args.penalty) or args.penalty <= 0:
        raise ValueError('penalty must be positive and finite')
    if not np.isfinite(args.eta) or not 0 < args.eta < 1:
        raise ValueError('eta must be finite and strictly between zero and one')
    if not np.isfinite(args.rmin) or args.rmin <= 0:
        raise ValueError('rmin must be positive and finite')
    if args.oc_warmup < 0 or not np.isfinite(args.oc_move) or not 0 < args.oc_move <= 1:
        raise ValueError('Require nonnegative oc-warmup and 0 < oc-move <= 1')
    config = make_config(args.benchmark, penalty=args.penalty, rmin=args.rmin)
    mesh_tag = f'{config.nelx}x{config.nely}'
    source = PROJECT/'results'/f'simp_{args.benchmark}_{mesh_tag}'
    baseline_hashes = hashes(source)
    betas = [float(x) for x in args.betas.split(',')]
    if not betas or not all(np.isfinite(x) and x > 0 for x in betas) or any(b <= a for a,b in zip(betas, betas[1:])):
        raise ValueError('Betas must be finite, positive and strictly increasing')
    seed_path = args.seed or source/'solution.npz'
    seed_hash = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    output = create_output(args.output or PROJECT/'results'/(
        f'simp_projected_{args.benchmark}_{mesh_tag}_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    with np.load(source/'solution.npz', allow_pickle=False) as saved:
        original = {k: saved[k].copy() for k in saved.files}
    with np.load(seed_path, allow_pickle=False) as seed:
        for key in ['nodes', 'elements', 'force', 'fixed_dofs']:
            if not np.array_equal(seed[key], original[key]):
                raise ValueError('Seed geometry/boundary mismatch: '+key)
        raw = np.clip(seed['raw_density'], .001, 1.)
    run = {'method': 'SIMP with density filter and Heaviside projection',
           'benchmark': args.benchmark, 'config': asdict(config), 'betas': betas,
           'eta': args.eta, 'penalty': args.penalty, 'grayness_constraint': False,
           'initialization': 'warm start from saved raw density; re-equilibrated at each beta',
           'initial_source': str(seed_path), 'seed_sha256': seed_hash, 'original_result_hashes': baseline_hashes,
           'oc_initializer': {'max_iter':args.oc_warmup, 'move':args.oc_move, 'final_acceptance':'original full-space NLP'},
           'solver': {'tol': 1e-7, 'max_iter': 1200, 'max_cpu_time_per_stage': args.max_cpu_time,
                      'linear_solver': 'mumps', 'acceptable_iter': 0, 'bound_relax_factor': 0},
           'projection': '(tanh(beta*eta)+tanh(beta*(filtered_density-eta)))/(tanh(beta*eta)+tanh(beta*(1-eta)))',
           'status': 'running'}
    write_json(output/'run_config.json', run)
    print('RESULT_DIR='+str(output), flush=True)
    rows = []
    started = time.perf_counter()
    mesh = None
    final_physical = None
    try:
        for stage, beta in enumerate(betas, 1):
            directory = create_output(output/f'stage_{stage:02d}_beta_{beta:g}')
            print(f'STAGE {stage}/{len(betas)} START beta={beta:g} p={args.penalty:g} eta={args.eta:g}', flush=True)
            tick = time.perf_counter()
            model, data = build_projected_model(config, raw, beta, args.eta, args.penalty)
            mesh = data.mesh
            if args.oc_warmup:
                from projected_simp_oc_seed import warm_start
                raw, oc_trace = warm_start(data, raw, beta, args.eta, args.penalty, args.oc_warmup, args.oc_move)
                write_json(directory/'oc_initialization_history.json', oc_trace)
                np.savez_compressed(directory/'oc_initialization_only.npz', raw_density=raw)
                # Re-equilibrate with the ORIGINAL assembler, not the vectorized initializer.
                physical = project(apply_density_filter(data.filter_data, raw), beta, args.eta)
                free_u, _ = solve_linear_response(mesh, data.element_ke, stiffness_factors(config, physical, args.penalty))
                for e in model.E:
                    model.x[e].set_value(float(raw[e]))
                for i, value in enumerate(free_u):
                    model.u[i].set_value(float(value))
            if stage == 1:
                for key in ['nodes', 'elements', 'force', 'fixed_dofs']:
                    if not np.array_equal(getattr(mesh, key), original[key]):
                        raise RuntimeError('Original geometry or boundary mismatch: '+key)
            solver = create_ipopt_solver(SolverSettings(tol=1e-7, max_iter=1200))
            solver.options.update({'max_cpu_time': args.max_cpu_time, 'OF_acceptable_iter': 0,
                                   'OF_output_file': str(directory/'ipopt_live.log'),
                                   'bound_relax_factor': 0, 'bound_push': 1e-4, 'bound_frac': 1e-4})
            result = solver.solve(model, tee=False, load_solutions=False, logfile=str(directory/'ipopt.log'))
            if len(result.solution) == 0:
                raise RuntimeError('Solver returned no iterate: '+str(result.solver.termination_condition))
            model.solutions.load_from(result)
            raw = np.array([pyo.value(model.x[e]) for e in model.E])
            filtered = apply_density_filter(data.filter_data, raw)
            physical = project(filtered, beta, args.eta)
            u = np.zeros(mesh.total_dofs)
            u[mesh.free_dofs] = [pyo.value(model.u[i]) for i in model.F]
            factors = stiffness_factors(config, physical, args.penalty)
            matrix = assemble_free_stiffness(mesh, data.element_ke, factors)
            residual = float(np.max(np.abs(matrix@u[mesh.free_dofs]-mesh.force[mesh.free_dofs])))
            _, fe_u = solve_linear_response(mesh, data.element_ke, factors)
            compliance = float(mesh.force@u)
            fe_compliance = float(mesh.force@fe_u)
            fe_error = abs(fe_compliance-compliance)/max(1., abs(fe_compliance))
            volume = float(physical.mean())
            accepted = bool(check_optimal_termination(result) and residual < 1e-6 and fe_error < 1e-6
                            and volume <= .4000001 and physical.min() >= -1e-12 and physical.max() <= 1+1e-12)
            row = {'stage': stage, 'beta': beta, 'penalty': args.penalty, 'eta': args.eta, 'rmin': config.rmin,
                   'compliance': compliance, 'volume_fraction': volume,
                   'global_grayness': float(np.mean(4*physical*(1-physical))),
                   'equilibrium_residual_inf': residual, 'fixed_design_fe_compliance': fe_compliance,
                   'fe_compliance_relative_error': fe_error,
                   'termination_condition': str(result.solver.termination_condition),
                   'accepted': accepted, 'elapsed_seconds': time.perf_counter()-tick}
            np.savez_compressed(directory/'solution.npz', nodes=mesh.nodes, elements=mesh.elements,
                                element_dofs=mesh.element_dofs, force=mesh.force, fixed_dofs=mesh.fixed_dofs,
                                free_dofs=mesh.free_dofs, raw_density=raw, filtered_density=filtered,
                                physical_density=physical, displacement=u)
            write_json(directory/'summary.json', row)
            rows.append(row)
            with (output/'stage_history.csv').open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print('STAGE_RESULT '+json.dumps(row), flush=True)
            if not accepted:
                run['status'] = 'stopped_at_unaccepted_stage'
                break
            final_physical = physical.copy()
        else:
            run['status'] = 'completed'
        if run['status'] == 'completed':
            # New directory only. No writer targets the original result directory.
            (output/'solution.npz').write_bytes((directory/'solution.npz').read_bytes())
            if args.benchmark == 'cantilever':
                plot_comparison(original['physical_density'], final_physical, mesh, rows, output)
    except Exception as exc:
        run['status'] = 'error'
        run['error'] = repr(exc)
        raise
    finally:
        run['elapsed_seconds'] = time.perf_counter()-started
        run['original_results_unchanged'] = hashes(source) == baseline_hashes
        run['seed_unchanged'] = hashlib.sha256(seed_path.read_bytes()).hexdigest() == seed_hash
        run['stages'] = rows
        write_json(output/'summary.json', run)
        print('FINAL_STATUS '+json.dumps({'status': run['status'], 'elapsed_seconds': run['elapsed_seconds'],
                                         'old_results_unchanged': run['original_results_unchanged']}), flush=True)
    if run['status'] != 'completed' or not run['original_results_unchanged']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
