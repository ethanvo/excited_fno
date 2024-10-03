#!/usr/bin/env python3
from pyscf import gto, scf, mp, cc
from pyscf.cc.eom_rccsd import EOMIP, EOMEA, EOMEESinglet
import numpy as np
from functools import reduce

mol = gto.Mole()
mol.atom = '''
    O   0.0000000   0.0000000   0.1177930
    H   0.0000000   0.7554150   -0.4711740
    H   0.0000000   -0.7554150  -0.4711740
'''
mol.basis = 'ccpvtz'
mol.verbose = 7
mol.build()

mymf = scf.RHF(mol)
mymf = mymf.density_fit()
mymf.kernel()

mymp = mp.RMP2(mymf)
emp2, t2 = mymp.kernel()
dm = mymp.make_rdm1(t2=t2)

nocc = mymp.nocc
nmo = mymp.nmo
nvir = nmo - nocc
mo_energy = mymf.mo_energy
mo_coeff = mymf.mo_coeff

results = []
for nvir_act in range(1, nvir):
    frozen = list(range(nocc+nvir_act, nmo))
    mycc = cc.RCCSD(mymf, frozen=frozen)
    eris = mycc.ao2mo()
    eccsd, t1, t2 = mycc.kernel(eris=eris)

    myeom = EOMIP(mycc)
    ipe, ipv = myeom.kernel(eris=eris)
    myeom = EOMEA(mycc)
    eae, eav = myeom.kernel(eris=eris)
    myeom = EOMEESinglet(mycc)
    eee, eev = myeom.kernel(eris=eris)
    results.append((nvir_act, ipe, eae, eee))

mycc = cc.RCCSD(mymf)
eris = mycc.ao2mo()
eccsd, t1, t2 = mycc.kernel()
myeom = EOMIP(mycc)
ipe, ipv = myeom.kernel(eris=eris)
myeom = EOMEA(mycc)
eae, eav = myeom.kernel(eris=eris)
myeom = EOMEESinglet(mycc)
eee, eev = myeom.kernel(eris=eris)
results.append((nvir, ipe, eae, eee))

# Save the results
with open('water_canon.dat', 'w') as f:
    f.write('nvir_act ipe eae eee\n')
    for r in results:
        f.write('%d %f %f %f\n' % r)

# Print the results
for r in results:
    print('%d %f %f %f' % r)

