#!/usr/bin/env python3
#
# Usage:
#   python gcga.py
#
# Input  : sub-POSCAR (required), ini.db (optional — random init if absent)
# Output : opt.db, sorted*.xyz, gcga.db
#
# To stop cleanly: touch STOP

import ase, ase.io
from ase.optimize import BFGS
from ase.build import bulk

from gocia.ga.popGrandCanon import PopulationGrandCanonical
from gocia.utils.ase import geomopt_iterate
from gocia.utils.dbio import get_traj
from gocia.ensemble import clusterIsomerAbs

from time import sleep
import os
import numpy as np

os.environ['HF_TOKEN'] = '(YOURTOKEN)'

AG_RADIUS   = 1.447   # Å
MAX_N       = 50      # largest cluster size expected
FMAX_INIT   = 0.05   # fmax for population seeding (fast)
FMAX_FINE   = 0.02   # fmax for GA main loop (precise)
DELTA_MU    = 0.0    # eV, shift from bulk E_Ag (positive → favor smaller clusters)

from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
my_calc   = FAIRChemCalculator(predictor, task_name="omat")

# --- Substrate relaxation ---
base = ase.io.read("sub-POSCAR")
base.calc = my_calc
dyn = BFGS(base, logfile='base.log', trajectory='base.traj')
dyn.run(fmax=0.02, steps=300)

print(os.getpid())

# --- Ag bulk chemical potential (computed with UMA) ---
ag_bulk = bulk('Ag', 'fcc', a=4.09) * (2, 2, 2)
ag_bulk.calc = my_calc
BFGS(ag_bulk, logfile=None).run(fmax=0.02, steps=300)
E_Ag = ag_bulk.get_potential_energy() / len(ag_bulk) + DELTA_MU
print(f'[*] E_Ag (UMA) = {E_Ag:.6f} eV/atom  (delta_mu={DELTA_MU:+.3f} eV)')

# --- z limits from substrate surface ---
z_surf  = base.positions[:, 2].max()
r_max   = (MAX_N ** (1/3)) * AG_RADIUS * 1.2   # radius of largest expected cluster
zLim    = [z_surf + 1.0, z_surf + r_max * 2 + 2.0]
print(f'[*] z_surf     = {z_surf:.3f} Å')
print(f'[*] r_max(N={MAX_N}) = {r_max:.2f} Å')
print(f'[*] zLim       = {zLim[0]:.2f} ~ {zLim[1]:.2f} Å')

cell = base.get_cell()
lx   = np.linalg.norm(cell[0])
ly   = np.linalg.norm(cell[1])

# --- GA parameters ---
totConf = 3000
minConf = 1000
popSize = 30
subsPot = base.get_potential_energy()

chemPotDict = {'Ag': E_Ag}

# --- Population setup ---
print('read in population')
pop = PopulationGrandCanonical(
    gadb         = 'gcga.db',
    substrate    = 'sub-POSCAR',
    popSize      = popSize,
    convergeCrit = popSize * 10,
    subsPot      = subsPot,
    chemPotDict  = chemPotDict,
    zLim         = zLim,
)

try:
    list_file = os.listdir('.')
    list_job  = [f for f in list_file if f[0] == 's' and '.' not in f]
    list_job.remove('sub-POSCAR')
    list_jid  = [int(j[1:]) for j in list_job]
    kidnum    = max(list_jid)
    print(f'Restarting the search from kidnum = {kidnum}')
except:
    kidnum = 0
    print('New search! Starting from kidnum = 0')

# --- Initialize population from ini.db (or random if not found) ---
if kidnum == 0:
    ini_db_path = 'ini.db'
    if os.path.exists(ini_db_path):
        print(f'Seeding population from ini.db...')
        from ase.db import connect as ase_connect
        ini_db = ase_connect(ini_db_path)
        cwd    = os.getcwd()

        for row in ini_db.select():
            kidnum += 1
            kiddir = f's{str(kidnum).zfill(6)}'
            os.makedirs(kiddir, exist_ok=True)
            os.chdir(kiddir)

            atoms   = row.toatoms()
            kid_opt = geomopt_iterate(
                atoms, my_calc,
                fmax        = FMAX_INIT,
                relax_steps = 200,
                substrate   = '../sub-POSCAR',
            )
            f64 = np.asarray(kid_opt.get_forces(), dtype=np.float64)
            if kid_opt.calc is not None:
                kid_opt.calc.results["forces"] = f64
            with open('label', 'w') as lf:
                lf.write(kiddir)
            os.chdir(cwd)

            with ase_connect('opt.db') as opt_db:
                opt_db.write(kid_opt, eV=kid_opt.get_potential_energy(), done=1)
            pop.add_aseResult(kid_opt, workdir=kiddir)

        # Deduplicate and sort → sorted*.xyz
        with ase_connect('opt.db') as raw_db:
            traj = get_traj(raw_db.select())
        clusterIsomerAbs(traj, eneToler=0.05, geomToler1=1e-3, geomToler2=0.5, outName='sorted')

        pop.natural_selection()
        print(f'Initialization done: {kidnum} structures → opt.db + sorted*.xyz')
    else:
        print('ini.db not found — using random initialization')
        pop.initializeDB()
        pop.natural_selection()

# --- Main GA loop ---
while not pop.is_converged() or kidnum < minConf:
    if 'STOP' in os.listdir():
        print('STOP REQUESTED')
        exit()
    if kidnum > totConf:
        print('MAX # SAMPLE REACHED')
        exit()

    # Generate offspring
    kidnum += 1
    kiddir = f's{str(kidnum).zfill(6)}'
    cwd    = os.getcwd()
    try:
        os.mkdir(kiddir)
    except:
        pass
    os.chdir(kiddir)

    kid = None
    while kid is None:
        kid = pop.gen_offspring_box(
            mutRate  = 0.4,
            xyzLims  = np.array([[0, lx], [0, ly], zLim]),
            transVec = [[-1,1,-2,2,-3,3,-4,4,-6,6,-12,12],
                        [-1,1,-2,2,-3,3,-6,6]],
        )
    kid.preopt_hooke(cutoff=1.2, toler=0.1)
    kid.print()
    kid.write('POSCAR')

    # Geometry optimization
    kid_opt = geomopt_iterate(
        kid.get_allAtoms(), my_calc,
        fmax        = FMAX_FINE,
        relax_steps = 250,
        substrate   = '../sub-POSCAR',
    )
    f64 = np.asarray(kid_opt.get_forces(), dtype=np.float64)
    if kid_opt.calc is not None:
        kid_opt.calc.results["forces"] = f64
    with open('label', 'w') as lf:
        lf.write(kiddir)

    # Update population
    os.chdir(cwd)
    pop.add_aseResult(kid_opt, workdir=kiddir)
    pop.natural_selection()
    sleep(1)

if pop.is_converged():
    print('CONVERGED!')
else:
    print('TERMINATED!')
