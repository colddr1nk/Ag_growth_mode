# Usage:
#   python workflow.py
#
# Input  : Ag.cif (in current directory)
# Output : 1_Ag_bulk_relaxed.vasp
#          2_Ag_slab.vasp
#          3_Ag_slab_supercell.vasp
#          4_Ag_slab_supercell_opt.vasp
#
# After running: manually generate sub-POSCAR and all-POSCAR,
# then copy them to 02 initial_structure/ and 03 gcga/

from ase.io import read, write
from ase.build import surface
from ase.build.supercells import make_supercell
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

import numpy as np
import spglib

import os
os.environ['HF_TOKEN'] = '(YOURTOKEN)'

atoms = read("Ag.cif")

from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms.calc = calc

cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
spglib.get_symmetry(cell, symprec=1e-5)
dataset = spglib.get_symmetry(cell, symprec=1e-5)

rotations = dataset['rotations']
translations = dataset['translations']

from ase.constraints import FixSymmetry
FixSymmetry(atoms, symprec=1e-5)
atoms.set_constraint(FixSymmetry(atoms, symprec=1e-5))

cell_filter = FrechetCellFilter(atoms)
opt = BFGS(cell_filter)
opt.run(fmax=0.01)

cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
print("Space group number:", dataset['number'])
print("International symbol:", dataset['international'])
print("Hall symbol:", dataset['hall'])

atoms.write("1_Ag_bulk_relaxed.vasp", format="vasp")


##### slab generation

atoms = read("1_Ag_bulk_relaxed.vasp")

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import surface

patoms = AseAtomsAdaptor.get_structure(atoms)

slab111 = surface.SlabGenerator(patoms,
                                miller_index=[1,1,1],
                                min_slab_size=8, min_vacuum_size=20,
                                lll_reduce=True, max_normal_search=10,
                                reorient_lattice=True).get_slabs(ftol=0.0001, symmetrize=True)
s111 = []
for a in slab111:
    s111.append(AseAtomsAdaptor.get_atoms(a))

slab = s111[0]
args = np.argsort(slab.positions[:,2])
slab = slab[args]
del slab[-1]

from ase.constraints import FixAtoms

slab.set_constraint(FixAtoms(indices=[0, 1]))

write("2_Ag_slab.vasp", slab, format='vasp')

## supercell

sslab = slab * (12,12,1) #크기 조절 필요

write("3_Ag_slab_supercell.vasp", sslab, format='vasp')

sslab.calc = calc
dyn = BFGS(sslab)
dyn.run(fmax=0.02)

write("4_Ag_slab_supercell_opt.vasp", sslab, format='vasp')

