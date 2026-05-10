# Usage:
#   python opt.py
#
# Input  : ini.db, sub-POSCAR (in current directory)
# Output : opt.db, sorted*.xyz

from ase.io import read, write

from ase.db import connect

from gocia.utils.ase import geomopt_iterate

from gocia.utils.dbio import get_traj, vasp2db, get_projName
from gocia.ensemble import clusterIsomerAbs

from ase.constraints import FixAtoms
import numpy as np

import os
os.environ['HF_TOKEN'] = '(YOURTOKEN)'

structs_raw = read('ini.db', index=':')
structs_sub = read('sub-POSCAR')

for atoms in structs_raw:
    cons = FixAtoms(indices=[atom.index for atom in atoms if atom.position[2] < 4])
    atoms.set_constraint(cons)

structs_opt = []

from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
my_calc = FAIRChemCalculator(predictor, task_name="omat")

for i in range(len(structs_raw)):
    my_label = f's{str(i).zfill(6)}'

    # OPTIMIZE THE STRUCTURE

    s_new = geomopt_iterate(structs_raw[i], my_calc, fmax=0.02, relax_steps=3000, label=my_label, substrate='sub-POSCAR')

    f64 = np.asarray(s_new.get_forces(), dtype=np.float64)
    if s_new.calc is not None:
        s_new.calc.results["forces"] = f64

    with connect('opt.db') as opt_db:
        opt_db.write(
            s_new,
            eV = s_new.get_potential_energy(),
            done=1,
        )

with connect('opt.db') as rawDB:
    traj = get_traj(rawDB.select())
    clusterIsomerAbs(
        traj,
        eneToler=0.05,
        geomToler1=1e-3,
        geomToler2=0.5,
        outName='sorted'
    )

