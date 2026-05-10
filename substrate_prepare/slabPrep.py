
from ase.io import read, write
from ase.build import sort
from ase.atoms import Atoms
import sys

inp = sys.argv[1]
z_fixNbuf = eval(sys.argv[2])
z_bufNads = eval(sys.argv[3])

inpSlab = sort(read(inp))
outSlab = inpSlab.copy()

del outSlab[[a.index for a in inpSlab if a.position[2] > z_fixNbuf]]

tmp = inpSlab.copy()
del tmp[[a.index for a in tmp if a.position[2] < z_fixNbuf or a.position[2] > z_bufNads ]]
outSlab.extend(tmp)
write('sub-'+inp, outSlab)

tmp = inpSlab.copy()
del tmp[[a.index for a in tmp if a.position[2] < z_bufNads]]
outSlab.extend(tmp)

write('all-'+inp, outSlab)
