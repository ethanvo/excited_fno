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
mymf.kernel()

mymp = mp.RMP2(mymf)
emp2, t2 = mymp.kernel()
dm = mymp.make_rdm1(t2=t2)

nocc = mymp.nocc
nmo = mymp.nmo
nvir = nmo - nocc
mo_energy = mymf.mo_energy
mo_coeff = mymf.mo_coeff

# 
mypt = cc.rccsd.RCCSD(mymf)
eris = mypt.ao2mo()
emp2, _, t2 = mypt.kernel(eris=eris, mbpt2=True)

myeom = EOMIP(mypt)
myeom.partition = 'mp'
eomipmpe, ipv = myeom.kernel(eris=eris, partition='mp')

myeom = EOMEA(mypt)
myeom.partition = 'mp'
eomeampe, eav = myeom.kernel(eris=eris, partition='mp')

myeom = EOMEESinglet(mypt)
myeom.partition = 'mp'
eomeempe, eev = myeom.kernel(eris=eris)

results = []
for nvir_act in range(1, nvir):
    n, v = np.linalg.eigh(dm[nocc:, nocc:])
    idx = np.argsort(n)[::-1]
    n, v = n[idx], v[:, idx]
    fvv = np.diag(mo_energy[nocc:])
    fvv_no = reduce(np.dot, (v.T.conj(), fvv, v))
    _, v_canon = np.linalg.eigh(fvv[:nvir_act, :nvir_act])
    no_coeff_1 = reduce(np.dot, (mo_coeff[:, nocc:], v[:, :nvir_act], v_canon))
    no_coeff_2 = np.dot(mo_coeff[:, nocc:], v[:, nvir_act:])
    no_coeff = np.concatenate((mo_coeff[:, :nocc], no_coeff_1, no_coeff_2), axis=1)

    frozen = list(range(nocc+nvir_act, nmo))
    pt_no = mp.RMP2(mymf, frozen=frozen, mo_coeff=no_coeff)
    pt_no.verbose=0
    pt_no.kernel()
    delta_emp2 = mymp.e_corr - pt_no.e_corr

    # Calculate delta EOM-MP2
    mypt = cc.rccsd.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    mypt.verbose = 0
    eris = mypt.ao2mo()
    emp2, _, t2 = mypt.kernel(eris=eris, mbpt2=True)
    myeom = EOMIP(mypt)
    myeom.verbose = 0
    myeom.partition = 'mp'
    efnoomipmpe, ipv = myeom.kernel(eris=eris, partition='mp')
    myeom = EOMEA(mypt)
    myeom.verbose = 0
    myeom.partition = 'mp'
    efnoeomipmpe, eav = myeom.kernel(eris=eris, partition='mp')
    myeom = EOMEESinglet(mypt)
    myeom.verbose = 0
    myeom.partition = 'mp'
    efnoeomeempe, eev = myeom.kernel(eris=eris)
    delta_eomipmpe = eomipmpe - efnoomipmpe
    delta_eomeampe = eomeampe - efnoeomipmpe
    delta_eomeempe = eomeempe - efnoeomeempe

    mycc = cc.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    eris = mycc.ao2mo()
    eccsd, t1, t2 = mycc.kernel(eris=eris)

    myeom = EOMIP(mycc)
    ipe, ipv = myeom.kernel(eris=eris)
    myeom = EOMEA(mycc)
    eae, eav = myeom.kernel(eris=eris)
    myeom = EOMEESinglet(mycc)
    eee, eev = myeom.kernel(eris=eris)
    results.append((nvir_act, delta_emp2, delta_eomipmpe, ipe, ipe+delta_eomipmpe, delta_eomeampe, eae, eae+delta_eomeampe, delta_eomeempe, eee, eee+delta_eomeempe))

mycc = cc.RCCSD(mymf)
eris = mycc.ao2mo()
eccsd, t1, t2 = mycc.kernel()
myeom = EOMIP(mycc)
ipe, ipv = myeom.kernel(eris=eris)
myeom = EOMEA(mycc)
eae, eav = myeom.kernel(eris=eris)
myeom = EOMEESinglet(mycc)
eee, eev = myeom.kernel(eris=eris)
results.append((nvir, 0, 0, ipe, ipe, 0, eae, eae, 0, eee, eee))

# Save the results
with open('water_fno.dat', 'w') as f:
    f.write('nvir_act delta_emp2 delta_eomipmpe ipe ipe+delta_eomipmpe delta_eomeampe eae eae+delta_eomeampe delta_eomeempe eee eee+delta_eomeempe\n')
    for r in results:
        f.write('%d %f %f %f %f %f %f %f %f %f %f\n' % r)

# Print the results
for r in results:
    print('%d %f %f %f %f %f %f %f %f %f %f' % r)

