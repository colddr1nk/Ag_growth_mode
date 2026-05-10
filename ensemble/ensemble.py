#!/usr/bin/env python3
#
# Usage:
#   python ensemble.py <n_atoms> <n_samples> [output_file] [--seed N]
#   python ensemble.py --merge [--pattern "ensemble_N*_S*.xyz"] [--output ensemble.xyz]
#
# Examples:
#   python ensemble.py 10 5
#   python ensemble.py 10 5 my_output.xyz --seed 42
#   python ensemble.py --merge --output ensemble.xyz

import os
import argparse
import logging
import numpy as np
import torch
from collections import Counter, deque
from ase import Atoms, units
from ase.build import bulk
from ase.io import read, write
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

class SingleAtomFilter(logging.Filter):
    def filter(self, record):
        return "Single atom system detected" not in record.getMessage()

logging.getLogger().addFilter(SingleAtomFilter())

os.environ['HF_TOKEN'] = '(YOUR_TOKEN)'
MODEL_NAME  = 'uma-s-1p1'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
ELEMENT     = 'Ag'

AG_LATTICE  = 4.09
AG_RADIUS   = (AG_LATTICE / np.sqrt(2)) / 2
MIN_DIST    = 2.20
BOND_CUTOFF = 3.20
VACUUM      = 10.0

BH_TEMP_K         = 1500
BH_TEMP_EV        = BH_TEMP_K * units.kB
FMAX              = 0.05
RANDOM_SEED_RATIO = 0.5
MAX_PLACEMENT_TRIES = 20000
TOP_K             = 5    # max structures per N (lowest energy first)


def get_box_size(n):
    if n <= 1: return round(2 * AG_RADIUS + 2 * VACUUM, 1)
    return round(2 * (n ** (1/3)) * AG_RADIUS * 1.2 + 2 * VACUUM, 1)

def get_step_size(n):
    return float(np.clip(0.5 * n ** (-1/3), 0.10, 0.50))

def get_bfgs_steps(n):
    return max(200, int(50 * n ** (1/3)))

def get_dynamic_bh_steps(n):
    return int(30 + (n * 0.5))


_PREDICTOR = None
def get_calc():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = pretrained_mlip.get_predict_unit(MODEL_NAME, device=DEVICE)
    return FAIRChemCalculator(_PREDICTOR, task_name='omat')


def place_in_fixed_box(atoms, n_atoms):
    box = get_box_size(n_atoms)
    s = atoms.copy()
    s.cell = [box, box, box]
    s.pbc = False
    if len(s) > 0:
        s.positions += np.array([box / 2] * 3) - s.get_center_of_mass()
    return s

def make_fcc_seed(n_atoms, noise=0.05):
    if n_atoms <= 1:
        return place_in_fixed_box(Atoms('Ag', positions=[(0, 0, 0)], pbc=False), 1) if n_atoms == 1 else Atoms()
    repeat = max(3, int(np.ceil(n_atoms ** (1/3))) + 3)
    fcc = bulk('Ag', 'fcc', a=AG_LATTICE) * (repeat, repeat, repeat)
    com = fcc.get_center_of_mass()
    idx = np.argsort(np.linalg.norm(fcc.positions - com, axis=1))[:n_atoms]
    positions = fcc.positions[idx].copy() + np.random.uniform(-noise, noise, (n_atoms, 3))
    return place_in_fixed_box(Atoms(ELEMENT * n_atoms, positions=positions, pbc=False), n_atoms)

def make_random_compact_seed(n_atoms, min_dist=MIN_DIST):
    if n_atoms <= 1: return make_fcc_seed(n_atoms)
    r_sphere = (n_atoms ** (1/3)) * AG_RADIUS * 0.9
    positions = []
    tries = 0
    while len(positions) < n_atoms:
        tries += 1
        if tries > MAX_PLACEMENT_TRIES:
            return make_fcc_seed(n_atoms)
        u, cos = np.random.uniform(0, 1), np.random.uniform(-1, 1)
        phi = np.random.uniform(0, 2 * np.pi)
        r   = r_sphere * u ** (1/3)
        sin = np.sqrt(1 - cos ** 2)
        pos = np.array([r * sin * np.cos(phi), r * sin * np.sin(phi), r * cos])
        if all(np.linalg.norm(pos - p) >= min_dist for p in positions):
            positions.append(pos)
    return place_in_fixed_box(Atoms(ELEMENT * n_atoms, positions=np.array(positions), pbc=False), n_atoms)

def generate_pool(n_atoms, n_structs):
    n_random = max(1, int(n_structs * RANDOM_SEED_RATIO))
    n_fcc    = max(1, n_structs - n_random)
    pool = []
    for _ in range(n_fcc):
        try: pool.append(('fcc',    make_fcc_seed(n_atoms)))
        except Exception: pass
    for _ in range(n_random):
        try: pool.append(('random', make_random_compact_seed(n_atoms)))
        except Exception: pass
    return pool


def is_connected(atoms):
    if len(atoms) <= 1: return True
    pos   = atoms.get_positions()
    diffs = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    visited, queue = {0}, deque([0])
    while queue:
        cur = queue.popleft()
        for j in range(len(atoms)):
            if j not in visited and dists[cur, j] <= BOND_CUTOFF:
                visited.add(j)
                queue.append(j)
    return len(visited) == len(atoms)

def get_coord_pattern(atoms):
    pos = atoms.get_positions()
    coords = [int((np.linalg.norm(pos - p, axis=1) < BOND_CUTOFF).sum()) - 1 for p in pos]
    return tuple(sorted(coords, reverse=True))

def check_geometry(atoms):
    if len(atoms) < 2: return True
    pos   = atoms.get_positions()
    diffs = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    return dists.min() >= MIN_DIST

def is_duplicate(a1, e1, a2, e2, e_tol=0.02, dist_tol=0.03):
    if len(a1) != len(a2) or (e1 is not None and e2 is not None and abs(e1 - e2) > e_tol) or get_coord_pattern(a1) != get_coord_pattern(a2):
        return False
    if len(a1) > 1:
        if np.mean(np.abs(np.sort(a1.get_all_distances().flatten()) - np.sort(a2.get_all_distances().flatten()))) > dist_tol:
            return False
    return True


def safe_relax(atoms, n_atoms):
    try:
        BFGS(atoms, logfile=None).run(fmax=FMAX, steps=get_bfgs_steps(n_atoms))
        E = atoms.get_potential_energy()
        if not np.isfinite(E) or abs(E / max(n_atoms, 1)) > 5.0: return None
        return E
    except Exception: return None

def collect_structure(atoms, energy, n_atoms):
    if not is_connected(atoms): return None
    s = place_in_fixed_box(atoms, n_atoms)
    s.info['energy'] = float(energy)
    return s


def merge_ensemble_files(pattern, output_file):
    import glob
    from collections import defaultdict

    files = sorted(glob.glob(pattern))
    if not files:
        print(f'[merge] No files matched: {pattern}')
        return

    print(f'[merge] {len(files)} files found')
    by_n = defaultdict(list)

    for f in files:
        try:
            structs = read(f, ':')
            for s in structs:
                try:
                    e = float(s.get_potential_energy())
                except Exception:
                    e = s.info.get('energy', None)
                if e is not None:
                    by_n[len(s)].append((e, s))
        except Exception as ex:
            print(f'  [!] skipped {f}: {ex}')

    merged = []
    print(f'\n{"N":>4} {"total":>7} {"kept":>6}  {"best E/N":>10}')
    print('-' * 34)
    for n in sorted(by_n.keys()):
        pairs    = sorted(by_n[n], key=lambda x: x[0])
        selected = pairs[:TOP_K]
        merged.extend(s for _, s in selected)
        print(f'{n:>4} {len(pairs):>7} {len(selected):>6}  {selected[0][0]/n:>10.4f}')

    write(output_file, merged, format='extxyz')
    print(f'\n[merge] {len(merged)} structures → {output_file}')


def generate_ag_ensemble(n_atoms, sampling_particles_count, output_file):
    print(f"--- Loading MLIP Model ({MODEL_NAME}) ---")
    get_calc()
    print(f"--- Model Loaded on {DEVICE} ---")
    full_structs = []

    for n in [n_atoms]:
        print(f"--- Starting Basin-Hopping for N={n} ---")
        pool = generate_pool(n, sampling_particles_count)
        if not pool: continue

        all_s, all_e       = [], []
        current_bh_steps   = get_dynamic_bh_steps(n)

        for _, (stype, seed) in enumerate(pool):
            try:
                atoms      = seed.copy()
                atoms.calc = get_calc()
                E_cur      = safe_relax(atoms, n)
                if E_cur is None: continue

                s = collect_structure(atoms, E_cur, n)
                if s: all_s.append(s); all_e.append(E_cur)

                for _ in range(current_bh_steps):
                    try:
                        cand = atoms.copy()
                        cand.positions += np.random.uniform(-get_step_size(n), get_step_size(n), cand.positions.shape)
                        if not check_geometry(cand): continue
                        cand.calc = get_calc()
                        E_new = safe_relax(cand, n)
                        if E_new is None: continue

                        if E_new - E_cur <= 0 or np.random.uniform() < np.exp(-(E_new - E_cur) / BH_TEMP_EV):
                            atoms, E_cur = cand.copy(), E_new
                            atoms.calc   = get_calc()

                        s = collect_structure(cand, E_new, n)
                        if s: all_s.append(s); all_e.append(E_new)
                    except Exception: continue
            except Exception: pass

        if not all_s: continue

        # Deduplicate
        unique_s, unique_e = [all_s[0]], [all_e[0]]
        for c, e_c in zip(all_s[1:], all_e[1:]):
            if not any(is_duplicate(c, e_c, r, e_r) for r, e_r in zip(unique_s, unique_e)):
                unique_s.append(c)
                unique_e.append(e_c)

        # Sort by energy and keep top K
        sorted_pairs = sorted(zip(unique_e, unique_s), key=lambda x: x[0])
        unique_s     = [s for _, s in sorted_pairs[:TOP_K]]
        full_structs.extend(unique_s)
        write(output_file, full_structs, format='extxyz')

    return full_structs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_atoms',                  type=int, nargs='?')
    parser.add_argument('sampling_particles_count', type=int, nargs='?')
    parser.add_argument('output_file',              type=str, nargs='?', default=None)
    parser.add_argument('--seed',                   type=int, default=None)
    parser.add_argument('--merge',                  action='store_true')
    parser.add_argument('--pattern',                default='ensemble_N*_S*.xyz')
    parser.add_argument('--output',                 default='ensemble.xyz')

    args = parser.parse_args()

    if args.merge:
        merge_ensemble_files(args.pattern, args.output)
    else:
        if args.n_atoms is None or args.sampling_particles_count is None:
            parser.print_help()
        else:
            if args.seed is not None:
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
            output_name = args.output_file or f'ensemble_N{args.n_atoms}_S{args.sampling_particles_count}.xyz'
            generate_ag_ensemble(args.n_atoms, args.sampling_particles_count, output_name)
