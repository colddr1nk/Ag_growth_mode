#!/usr/bin/env python3
#
# Usage:
#   python ini.py <cluster_file> <n_samples>
#
# Example:
#   python ini.py ensemble.xyz 30
#
# Input  : <cluster_file> (merged ensemble xyz), sub-POSCAR
# Output : ini.db

import os
import sys
import random
import argparse
import numpy as np
from collections import defaultdict, Counter

from ase.io import read
from ase.db import connect
from ase.constraints import FixAtoms

SUB_POSCAR   = 'sub-POSCAR'
ALL_POSCAR   = None
OUTPUT_FILE  = 'ini.db'

TARGET_ATOMS_MIN    = 50      # min deposited atoms per sample
TARGET_ATOMS_MAX    = 250     # max deposited atoms per sample
MIN_CLUSTER_N       = 5       # exclude clusters smaller than this
MIN_ATOM_DIST       = 2.3     # Å, min substrate-cluster distance
MIN_CLUSTER_CLUSTER = 4.0     # Å, min cluster-cluster distance
MAX_COVERAGE        = 0.50    # max coverage fraction per cell area
ZLIM_MIN_OFFSET     = 1.5     # Å, anchor z min above surface
ZLIM_MAX_OFFSET     = 3.0     # Å, anchor z max above surface
MAX_PLACE_TRIES     = 200
MAX_LOOP_TRIES      = 1000
AG_RADIUS           = 1.447   # Å


def get_max_count(n_atoms, budget, slab_area):
    r    = (n_atoms ** (1/3)) * AG_RADIUS * 1.2
    area = np.pi * r ** 2
    return min(
        max(1, int(slab_area * MAX_COVERAGE / area)),
        max(1, budget // n_atoms)
    )


def _parse_fixatoms(poscar_file, n_atoms):
    fixed      = []
    atom_start = None
    with open(poscar_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith('selective'):
            atom_start = i + 2
            break
        if s.startswith('cartesian') or s.startswith('direct'):
            atom_start = i + 1
            break
    if atom_start is None:
        return []
    for idx in range(n_atoms):
        li = atom_start + idx
        if li >= len(lines):
            break
        parts = lines[li].split()
        if len(parts) >= 6 and parts[3:6] == ['F', 'F', 'F']:
            fixed.append(idx)
    return fixed


def load_substrate(sub_file, all_file=None):
    sub = read(sub_file, format='vasp')
    if all_file and os.path.exists(all_file):
        ref        = read(all_file, format='vasp')
        cell_match = np.allclose(sub.cell[:], ref.cell[:], atol=0.01)
        print(f'[*] all-POSCAR cell match: {cell_match}')
    n_fixed = sum(len(c.index) for c in sub.constraints if isinstance(c, FixAtoms))
    if n_fixed == 0:
        idx = _parse_fixatoms(sub_file, len(sub))
        if idx:
            sub.set_constraint(FixAtoms(indices=idx))
            n_fixed = len(idx)
    cell  = sub.get_cell()
    lx    = np.linalg.norm(cell[0])
    ly    = np.linalg.norm(cell[1])
    lz    = np.linalg.norm(cell[2])
    z_max = sub.positions[:, 2].max()
    print(f'[substrate] {len(sub)} atoms | {lx:.1f}x{ly:.1f}x{lz:.1f} Å | fixed {n_fixed} | surface z={z_max:.3f} Å')
    return sub


def load_clusters(cluster_file, min_n=MIN_CLUSTER_N):
    raw  = read(cluster_file, ':')
    by_n = defaultdict(list)
    skip = 0
    for a in raw:
        if len(a) < min_n:
            skip += 1
            continue
        s = a.copy()
        s.calc = None
        s.set_constraint()
        s.info.clear()
        s.positions -= s.get_center_of_mass()
        s.positions[:, 2] -= s.positions[:, 2].min()
        by_n[len(a)].append(s)
    ns = sorted(by_n.keys())
    print(f'[clusters] {sum(len(v) for v in by_n.values())} structures | N={min(ns)}~{max(ns)} | skipped(N<{min_n}) {skip}')
    return by_n


def min_dist_pbc(pos1, pos2, cell):
    diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
    lx   = np.linalg.norm(cell[0])
    ly   = np.linalg.norm(cell[1])
    diff[:, :, 0] -= np.round(diff[:, :, 0] / lx) * lx
    diff[:, :, 1] -= np.round(diff[:, :, 1] / ly) * ly
    return np.sqrt((diff ** 2).sum(axis=-1)).min()


def check_collision(slab, new_mol, cell, n_sub):
    pos_new = new_mol.get_positions()
    pos_sub = slab.get_positions()[:n_sub]
    pos_cl  = slab.get_positions()[n_sub:]
    if min_dist_pbc(pos_sub, pos_new, cell) < MIN_ATOM_DIST:
        return True
    if len(pos_cl) > 0 and min_dist_pbc(pos_cl, pos_new, cell) < MIN_CLUSTER_CLUSTER:
        return True
    return False


def random_rotation(atoms):
    s = atoms.copy()
    s.rotate(np.random.uniform(0, 360), 'z', center='COM')
    vec  = np.random.randn(3)
    vec[2] = 0.0
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        s.rotate(np.random.uniform(-30, 30), vec / norm, center='COM')
    s.positions[:, 2] -= s.positions[:, 2].min()  # keep bottom atom at z=0
    return s


def place_sample(cluster_lib, sub_ref, zlim):
    cell      = sub_ref.get_cell()
    lx        = np.linalg.norm(cell[0])
    ly        = np.linalg.norm(cell[1])
    slab_area = lx * ly
    n_sub     = len(sub_ref)
    slab      = sub_ref.copy()
    avail_ns  = sorted(cluster_lib.keys())
    budget    = random.randint(TARGET_ATOMS_MIN, TARGET_ATOMS_MAX)
    remaining = budget
    n_placed, sizes, count, attempts = 0, [], Counter(), 0

    while remaining > 0 and attempts < MAX_LOOP_TRIES:
        attempts += 1
        cands = [n for n in avail_ns
                 if n <= remaining and count[n] < get_max_count(n, budget, slab_area)]
        if not cands:
            break

        n_pick = random.choice(cands)
        mol    = random_rotation(random.choice(cluster_lib[n_pick]))
        placed = False

        for _ in range(MAX_PLACE_TRIES):
            r_margin = (n_pick ** (1/3)) * AG_RADIUS * 1.2
            rx  = random.uniform(r_margin, lx - r_margin)
            ry  = random.uniform(r_margin, ly - r_margin)
            rz  = random.uniform(zlim[0], zlim[1])
            tmp = mol.copy()
            tmp.positions += [rx, ry, rz]
            if not check_collision(slab, tmp, cell, n_sub):
                slab       = slab + tmp
                remaining -= n_pick
                n_placed  += 1
                sizes.append(n_pick)
                count[n_pick] += 1
                placed = True
                break

        if not placed:
            attempts += MAX_PLACE_TRIES // 10

    return slab, n_placed, sizes


def validate(slab, sub_ref):
    n_sub  = len(sub_ref)
    z_surf = sub_ref.positions[:, 2].max()
    cell   = slab.get_cell()
    pos    = slab.get_positions()
    diff   = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    lx     = np.linalg.norm(cell[0])
    ly     = np.linalg.norm(cell[1])
    diff[:, :, 0] -= np.round(diff[:, :, 0] / lx) * lx
    diff[:, :, 1] -= np.round(diff[:, :, 1] / ly) * ly
    d_mat  = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(d_mat, np.inf)
    min_d_all    = float(d_mat.min())
    min_d_sub_cl = float(d_mat[:n_sub, n_sub:].min()) if len(slab) > n_sub else np.inf
    cz    = slab.positions[n_sub:]
    z_gap = float(cz[:, 2].min()) - z_surf if len(cz) else 0.0
    return {
        'min_dist'        : min_d_all,
        'min_dist_sub_cl' : min_d_sub_cl,
        'z_gap'           : z_gap,
        'min_dist_ok'     : min_d_sub_cl >= MIN_ATOM_DIST,
    }


def main():
    parser = argparse.ArgumentParser(description='Deposit Ag nanoclusters on Ag(111) substrate')
    parser.add_argument('cluster_file', type=str)
    parser.add_argument('n_samples',    type=int)
    args = parser.parse_args()

    for f in [args.cluster_file, SUB_POSCAR]:
        if not os.path.exists(f):
            print(f'[Error] file not found: {f}')
            sys.exit(1)

    sub_ref     = load_substrate(SUB_POSCAR, ALL_POSCAR)
    cluster_lib = load_clusters(args.cluster_file)
    if not cluster_lib:
        print('[Error] no valid clusters found.')
        sys.exit(1)

    z_surf = sub_ref.positions[:, 2].max()
    zlim   = [z_surf + ZLIM_MIN_OFFSET, z_surf + ZLIM_MAX_OFFSET]

    print(f'\n[*] Deposition parameters')
    print(f'    zLim         : {zlim[0]:.2f} ~ {zlim[1]:.2f} Å')
    print(f'    atom budget  : {TARGET_ATOMS_MIN}~{TARGET_ATOMS_MAX}')
    print(f'    max coverage : {MAX_COVERAGE:.0%}')
    print(f'    sub-cluster  : {MIN_ATOM_DIST} Å')
    print(f'    cl-cl        : {MIN_CLUSTER_CLUSTER} Å')
    print(f'    n_samples    : {args.n_samples}')

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    db = connect(OUTPUT_FILE)

    print(f'\n--- Generating {args.n_samples} samples ---')
    n_ok, all_sizes, all_n_atoms = 0, [], []

    for i in range(args.n_samples):
        slab, n_placed, sizes = place_sample(cluster_lib, sub_ref, zlim)

        if n_placed == 0:
            print(f'  [!] Sample {i+1:02d}: placement failed, skipping.')
            continue

        val = validate(slab, sub_ref)
        if not val['min_dist_ok']:
            print(f'  [!] Sample {i+1:02d}: distance check failed '
                  f'(sub-cl {val["min_dist_sub_cl"]:.3f}/{MIN_ATOM_DIST}Å), skipping.')
            continue

        n_ca      = len(slab) - len(sub_ref)
        sizes_str = '[' + ','.join(map(str, sorted(sizes))) + ']'
        all_sizes.extend(sizes)
        all_n_atoms.append(n_ca)

        db.write(
            slab,
            sample_id       = i + 1,
            n_clusters      = n_placed,
            n_cluster_atoms = n_ca,
            n_total_atoms   = len(slab),
            min_dist        = round(val['min_dist'], 4),
            z_gap           = round(val['z_gap'], 3),
            cluster_sizes   = sizes_str,
            substrate       = os.path.basename(SUB_POSCAR),
        )
        n_ok += 1
        print(f'  [+] Sample {i+1:02d}: {n_placed} clusters {sizes_str} '
              f'| cluster atoms {n_ca} | min_d {val["min_dist"]:.3f}Å | z_gap {val["z_gap"]:.2f}Å')

    print(f'\n{"="*60}')
    print(f'  Done: {n_ok}/{args.n_samples} → {OUTPUT_FILE}')
    print(f'{"="*60}')
    if n_ok > 0:
        print(f'  deposited atoms: avg {np.mean(all_n_atoms):.1f} ({min(all_n_atoms)}~{max(all_n_atoms)})')
        print(f'  N distribution : {dict(sorted(Counter(all_sizes).items()))}')


if __name__ == '__main__':
    main()
