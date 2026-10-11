"""Export only the last accepted projected-SIMP stage, without replacing any files."""
from pathlib import Path
import json
import shutil
import numpy as np
from run_simp_projected_paper import create_output, plot_comparison, hashes
from tpo.simp_model import SimpConfig, build_benchmark_mesh

run_dir = Path('D:/Case/TPO/results/simp_projected_cantilever_60x20_20260909_181431')
run = json.loads((run_dir/'summary.json').read_text(encoding='utf-8'))
accepted = [row for row in run['stages'] if row['accepted']]
last = accepted[-1]
if last['beta'] != 14 or last['termination_condition'] != 'optimal':
    raise RuntimeError('Unexpected last converged stage; review before exporting')
source = run_dir/f"stage_{last['stage']:02d}_beta_{last['beta']:g}"
output = create_output(run_dir/'converged_beta14')
for name in ['solution.npz', 'summary.json']:
    shutil.copyfile(source/name, output/name)
old_dir = Path('D:/Case/TPO/results/simp_cantilever_60x20')
with np.load(old_dir/'solution.npz', allow_pickle=False) as archive:
    old = archive['physical_density'].copy()
with np.load(output/'solution.npz', allow_pickle=False) as archive:
    new = archive['physical_density'].copy()
config = SimpConfig(nelx=60, nely=20, volfrac=.4, rmin=1.5, penalties=(3.,))
plot_comparison(old, new, build_benchmark_mesh(config), accepted, output)
check = {'original_results_unchanged': hashes(old_dir) == run['original_result_hashes'],
         'export_matches_stage': (output/'solution.npz').read_bytes() == (source/'solution.npz').read_bytes(),
         'volume_fraction_recomputed': float(new.mean()),
         'grayness_recomputed': float(np.mean(4*new*(1-new))),
         'selected_beta': 14, 'beta16_converged': False}
if not check['original_results_unchanged'] or not check['export_matches_stage']:
    raise RuntimeError('Result preservation check failed')
(output/'export_check.json').write_text(json.dumps(check, indent=2), encoding='utf-8')
print(json.dumps({'output': str(output), **check}, ensure_ascii=False))
