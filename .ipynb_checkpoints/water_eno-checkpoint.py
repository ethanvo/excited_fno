#!/usr/bin/env python3
from pyscf import gto, scf, mp, cc
from pyscf.cc.eom_rccsd import EOMIP, EOMEA, EOMEESinglet
import numpy as np
from functools import reduce

def make_fno(dm, mo_energy, mo_coeff, nvir_act):
    n, v = np.linalg.eigh(dm)
    idx = np.argsort(n)[::-1]
    n, v = n[idx], v[:, idx]
    fvv = np.diag(mo_energy[nocc:])
    fvv_no = reduce(np.dot, (v.T.conj(), fvv, v))
    _, v_canon = np.linalg.eigh(fvv[:nvir_act, :nvir_act])
    no_coeff_1 = reduce(np.dot, (mo_coeff[:, nocc:], v[:, :nvir_act], v_canon))
    no_coeff_2 = np.dot(mo_coeff[:, nocc:], v[:, nvir_act:])
    no_coeff = np.concatenate((mo_coeff[:, :nocc], no_coeff_1, no_coeff_2), axis=1)
    return no_coeff

def get_delta_emp2(mymp, mymf, frozen, no_coeff): 
    pt_no = mp.RMP2(mymf, frozen=frozen, mo_coeff=no_coeff)
    pt_no.verbose=0
    pt_no.kernel()
    delta_emp2 = mymp.e_corr - pt_no.e_corr
    return delta_emp2

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

mymp = cc.rccsd.RCCSD(mymf)
eris = mymp.ao2mo()
emp2, _, t2 = mymp.kernel(eris=eris, mbpt2=True)
nocc = mymp.nocc
nmo = mymp.nmo
nvir = nmo - nocc
mo_energy = mymf.mo_energy
mo_coeff = mymf.mo_coeff

myeom = EOMIP(mymp)
myeom.partition = 'mp'
eomipmpe, ipv = myeom.kernel(eris=eris, partition='mp')
r1, r2 = myeom.vector_to_amplitudes(ipv, nmo, nocc)
l2 = r2.conj()
ipdm  = 2 * np.einsum('ija,ijb->ba', l2, r2)
ipdm -=     np.einsum('ija,ijb->ba', l2, r2)

myeom = EOMEA(mymp)
myeom.partition = 'mp'
eomeampe, eav = myeom.kernel(eris=eris, partition='mp')
r1, r2 = myeom.vector_to_amplitudes(eav, nmo, nocc)
l2 = r2.conj()
eadm  = 2 * np.einsum('ica,icb->ba', l2, r2)
eadm -=     np.einsum('ica,ibc->ba', l2, r2)

myeom = EOMEESinglet(mymp)
myeom.partition = 'mp'
eomeempe, eev = myeom.kernel(eris=eris)
r1, r2 = myeom.vector_to_amplitudes(eev, nmo, nocc)
l2 = r2.conj()
eedm  = 2 * np.einsum('ijca,ijcb->ba', l2, r2)
eedm -=     np.einsum('ijca,ijbc->ba', l2, r2)

results = []
for nvir_act in range(1, nvir):
    frozen = list(range(nocc+nvir_act, nmo))
    no_coeff = make_fno(ipdm, mo_energy, mo_coeff, nvir_act)
    ip_delta_emp2 = get_delta_emp2(mymp, mymf, frozen, no_coeff)
    # Calculate delta EOM-MP2 correction
    mypt = cc.rccsd.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    mypt.verbose = 0
    eris = mypt.ao2mo()
    emp2, _, t2 = mypt.kernel(eris=eris, mbpt2=True)
    myeom = EOMIP(mypt)
    myeom.verbose = 0
    myeom.partition = 'mp'
    eenoomipmpe, ipv = myeom.kernel(eris=eris, partition='mp')
    delta_eomipmpe = eomipmpe - eenoomipmpe
    mycc = cc.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    eris = mycc.ao2mo()
    eccsd, t1, t2 = mycc.kernel(eris=eris)
    myeom = EOMIP(mycc)
    ipe, ipv = myeom.kernel(eris=eris)

    no_coeff = make_fno(eadm, mo_energy, mo_coeff, nvir_act)
    ea_delta_emp2 = get_delta_emp2(mymp, mymf, frozen, no_coeff)
    # Calculate delta EOM-MP2 correction
    mypt = cc.rccsd.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    mypt.verbose = 0
    eris = mypt.ao2mo()
    emp2, _, t2 = mypt.kernel(eris=eris, mbpt2=True)
    myeom = EOMEA(mypt)
    myeom.verbose = 0
    myeom.partition = 'mp'
    eenoomeampe, eav = myeom.kernel(eris=eris, partition='mp')
    delta_eomeampe = eomeampe - eenoomeampe
    mycc = cc.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    eris = mycc.ao2mo()
    eccsd, t1, t2 = mycc.kernel(eris=eris)
    myeom = EOMEA(mycc)
    eae, eav = myeom.kernel(eris=eris)

    no_coeff = make_fno(eedm, mo_energy, mo_coeff, nvir_act)
    ee_delta_emp2 = get_delta_emp2(mymp, mymf, frozen, no_coeff)
    # Calculate delta EOM-MP2 correction
    mypt = cc.rccsd.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    mypt.verbose = 0
    eris = mypt.ao2mo()
    emp2, _, t2 = mypt.kernel(eris=eris, mbpt2=True)
    myeom = EOMEESinglet(mypt)
    myeom.verbose = 0
    eenoeempe, eev = myeom.kernel(eris=eris)
    delta_eomeempe = eomeempe - eenoeempe
    mycc = cc.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    eris = mycc.ao2mo()
    eccsd, t1, t2 = mycc.kernel(eris=eris)
    myeom = EOMEESinglet(mycc)
    eee, eev = myeom.kernel(eris=eris)
    results.append((nvir_act, ip_delta_emp2, delta_eomipmpe, ipe, ipe+delta_eomipmpe, ea_delta_emp2, delta_eomeampe, eae, eae+delta_eomeampe, ee_delta_emp2, delta_eomeempe, eee, eee+delta_eomeempe))

mycc = cc.RCCSD(mymf)
eris = mycc.ao2mo()
eccsd, t1, t2 = mycc.kernel()
myeom = EOMIP(mycc)
ipe, ipv = myeom.kernel(eris=eris)
myeom = EOMEA(mycc)
eae, eav = myeom.kernel(eris=eris)
myeom = EOMEESinglet(mycc)
eee, eev = myeom.kernel(eris=eris)
results.append((nvir, 0, 0, ipe, ipe, 0, 0, eae, eae, 0, 0, eee, eee))

# Save the results
with open('water_eno.dat', 'w') as f:
    f.write('nvir_act ip_delta_emp2 delta_eomipmpe ipe ipe+delta_eomipmpe ea_delta_emp2 delta_eomeampe eae eae+delta_eomeampe ee_delta_emp2 delta_eomeempe eee eee+delta_eomeempe\n')
    for r in results:
        f.write('%d %f %f %f %f %f %f %f %f %f %f %f %f\n' % r)

# Print the results
for r in results:
    print('%d %f %f %f %f %f %f %f %f %f %f %f %f' % r)

