#!/usr/bin/env python3
from pyscf import gto, scf, mp, cc
from pyscf.tdscf.rhf import CIS
from pyscf.cc.eom_rccsd import EOMIP, EOMEA, EOMEESinglet
import numpy as np
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
import berkelplot
from copy import deepcopy

einsum = np.einsum

def make_fno(dm, mo_energy, mo_coeff, nocc, nvir_act):
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

def ip_eom_dm(t1, t2, r1, r2, l1, l2):
    lab = 0.5 * einsum('ija,ijb->ab', l2, r2)
    y1ia = einsum('ija,j->ia', l2, r1)
    yab  = 0.5 * (lab - einsum('ia,ib->ab', y1ia, t1))
    yab += 0.5 * (lab.transpose(1, 0) - einsum('ib,ia->ab', y1ia, t1))
    return yab

def ea_eom_dm(t1, t2, r1, r2, l1, l2):
    lab = einsum('iac,ibc->ab', l2, r2)
    y1ia = einsum('iac,c->ia', l2, r1)
    yab  = 0.5 * (einsum('a,b->ab', l1, r1) + lab - einsum('ia,ib->ab', y1ia, t1))
    yab += 0.5 * (einsum('b,a->ab', l1, r1) + lab.transpose(1, 0) - einsum('ib,ia->ab', y1ia, t1))
    return yab

def eomipccsd(mymf, frozen, no_coeff):
    mycc = cc.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    eris = mycc.ao2mo()
    eccsd, t1, t2 = mycc.kernel(eris=eris)
    myeom = EOMIP(mycc)
    ipe, ipv = myeom.kernel(eris=eris)
    return ipe, ipv

def eomeaccsd(mymf, frozen, no_coeff):
    mycc = cc.RCCSD(mymf, frozen=frozen, mo_coeff=no_coeff)
    eris = mycc.ao2mo()
    eccsd, t1, t2 = mycc.kernel(eris=eris)
    myeom = EOMEA(mycc)
    eae, eav = myeom.kernel(eris=eris)
    return eae, eav

def ip_eom_tdm(t1, t2, r1, r2, l1, l2):
    # Intermediates
    y1ia = einsum('ija,j->ia', l2, r1)
    la   = 0.5 * einsum('ijb,ijab->a', l2, t2)
    lij  = einsum('ika,jka->ij', l2, r2)
    lab  = 0.5 * einsum('ija,ijb->ab', l2, r2)
    # Construct the TDM
    yia  = y1ia
    tmp  = einsum('i,ja->ija', r1, t1)
    tmp += r2
    yia += einsum('ija,j->ia', tmp, l1)
    tmp  = einsum('ja,ib->ijab', t1, t1)
    tmp -= t2
    yia += einsum('ijab,jb->ia', tmp, y1ia)
    yia -= einsum('i,a->ia', r1, la)
    yia -= einsum('ji,ja->ia', lij, t1)
    yia -= einsum('ba,ib->ia', lab, t1)
    yia += t1
    yia *= 0.5
    return yia

def ea_eom_tdm(t1, t2, r1, r2, l1, l2):
    # Intermediates
    y1ia = einsum('iac,c->ia', l2, r1)
    li   = 0.5 * einsum('kdc,ikcd->i', l2, t2)
    lij  = 0.5 * einsum('icd,jcd->ij', l2, r2)
    lab  = einsum('kac,kbc->ab', l2, r2)
    # Construct the TDM
    yia  = y1ia
    tmp  = einsum('ic,a->iac', t1, r1)
    tmp  = r2 - tmp
    yia += einsum('iac,c->ia', tmp, l1)
    tmp  = einsum('ka,ic->kica', t1, t1)
    tmp  = t2 - tmp
    yia += einsum('kica,kc->ia', tmp, y1ia)
    yia -= einsum('i,a->ia', li, r1)
    yia -= einsum('ka,ki->ia', t1, lij)
    yia -= einsum('ic,ca->ia', t1, lab)
    yia += t1
    yia *= 0.5
    return yia

def make_nto(mo_coeff, nocc, t1):
    orbo = mo_coeff[:, :nocc]
    orbv = mo_coeff[:, nocc:]
    nto_o, w, nto_vT = np.linalg.svd(t1)
    nto_v = nto_vT.conj().T
    idx = np.argmax(abs(nto_o.real), axis=0)
    nto_o[:, nto_o[idx,np.arange(nocc)].real<0] *= -1
    idx = np.argmax(abs(nto_v.real), axis=0)
    nto_v[:, nto_v[idx,np.arange(nvir)].real<0] *= -1

    occupied_nto = np.dot(orbo, nto_o)
    virtual_nto = np.dot(orbv, nto_v)
    nto_coeff = np.hstack((occupied_nto, virtual_nto))
    return nto_coeff

if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = '''
        C   0.0000000   1.3975060   0.0000000
        C   1.2102760   0.6987530   0.0000000
        C   1.2102760   -0.6987530  0.0000000
        C   0.0000000   -1.3975060  0.0000000
        C   -1.2102760  -0.6987530  0.0000000
        C   -1.2102760  0.6987530   0.0000000
        H   0.0000000   2.4806790   0.0000000
        H   2.1483310   1.2403390   0.0000000
        H   2.1483310   -1.2403390  0.0000000
        H   0.0000000   -2.4806790  0.0000000
        H   -2.1483310  -1.2403390  0.0000000
        H   -2.1483310  1.2403390   0.0000000
    '''
    mol.basis = 'ccpvtz'
    mol.verbose = 7
    mol.build()

    mymf = scf.RHF(mol)
    mymf.kernel()
    mo_energy = mymf.mo_energy
    mo_coeff = mymf.mo_coeff

    # MP DM

    mymp = mp.RMP2(mymf)
    emp2, t2 = mymp.kernel()
    mpdm = mymp.make_rdm1(t2=t2)

    # NTOs

    mycis = CIS(mymf)
    ecis, xy = mycis.kernel()
    weights, nto_coeff = mycis.get_nto()
    cis_t1 = xy[0][0]
    cis_t1 *= 1. / np.linalg.norm(cis_t1)

    # EOM-DM

    mymp = cc.rccsd.RCCSD(mymf)
    eris = mymp.ao2mo()
    emp2, t1, t2 = mymp.kernel(eris=eris, mbpt2=True)
    nocc = mymp.nocc
    nmo = mymp.nmo
    nvir = nmo - nocc

    myeom = EOMIP(mymp)
    myeom.partition = 'mp'
    eomipmpe, ipv = myeom.kernel(eris=eris, partition='mp')
    r1, r2 = myeom.vector_to_amplitudes(ipv, nmo, nocc)
    leomipmpe, lipv = myeom.kernel(left=True, eris=eris, partition='mp')
    l1, l2 = myeom.vector_to_amplitudes(lipv, nmo, nocc)
    ipdm  = ip_eom_dm(t1, t2, r1, r2, l1, l2)
    iptdm = ip_eom_tdm(t1, t2, r1, r2, l1, l2)
    ipeto_coeff = make_nto(mo_coeff, nocc, iptdm)
    ipento_coeff = make_nto(mo_coeff, nocc, iptdm + cis_t1)

    myeom = EOMEA(mymp)
    myeom.partition = 'mp'
    eomeampe, eav = myeom.kernel(eris=eris, partition='mp')
    r1, r2 = myeom.vector_to_amplitudes(eav, nmo, nocc)
    leomeampe, lav = myeom.kernel(left=True, eris=eris, partition='mp')
    l1, l2 = myeom.vector_to_amplitudes(lav, nmo, nocc)
    eadm  = ea_eom_dm(t1, t2, r1, r2, l1, l2)
    eatdm = ea_eom_tdm(t1, t2, r1, r2, l1, l2)
    eaeto_coeff = make_nto(mo_coeff, nocc, eatdm)
    eaento_coeff = make_nto(mo_coeff, nocc, eatdm + cis_t1)

    # Combined DMs
    mpdm = mpdm[nocc:, nocc:]
    mp_ip_dm = mpdm + ipdm
    mp_ea_dm = mpdm + eadm

    """
    myeom = EOMEESinglet(mymp)
    myeom.partition = 'mp'
    eomeempe, eev = myeom.kernel(eris=eris)
    r1, r2 = myeom.vector_to_amplitudes(eev, nmo, nocc)
    l2 = r2.conj()
    eedm  = 2 * np.einsum('ijca,ijcb->ba', l2, r2)
    eedm -=     np.einsum('ijca,ijbc->ba', l2, r2)
    """
    # Initialize data lists
    data = {'fnoip': [], 'fnoea': [], 'ntoip': [], 'ntoea': [], 'enoip': [], 'enoea': [], 'efnoip': [], 'efnoea': [], 'etoip': [], 'etoea': [], 'entoip': [], 'entoea': [], 'canip': [], 'canea': []}
    
    for nvir_act in range(1, nvir):
        # Test FNO, NTO, IPDM, EADM, MP_IP_DM, MP_EA_DM
        frozen = list(range(nocc+nvir_act, nmo))
        
        # FNO
        fno_coeff = make_fno(mpdm, mo_energy, mo_coeff, nocc, nvir_act)
        fno_ipe, fno_ipv = eomipccsd(mymf, frozen, fno_coeff)
        fno_eae, fno_eav = eomeaccsd(mymf, frozen, fno_coeff)

        # NTO
        nto_ipe, nto_ipv = eomipccsd(mymf, frozen, nto_coeff)
        nto_eae, nto_eav = eomeaccsd(mymf, frozen, nto_coeff)

        # ENO
        enoip_coeff = make_fno(ipdm, mo_energy, mo_coeff, nocc, nvir_act)
        enoip_ipe, enoip_ipv = eomipccsd(mymf, frozen, enoip_coeff)
        enoea_coeff = make_fno(eadm, mo_energy, mo_coeff, nocc, nvir_act)
        enoea_eae, enoea_eav = eomeaccsd(mymf, frozen, enoea_coeff)

        # EFNO
        efnoip_coeff = make_fno(mp_ip_dm, mo_energy, mo_coeff, nocc, nvir_act)
        efnoip_ipe, efnoip_ipv = eomipccsd(mymf, frozen, efnoip_coeff)
        efnoea_coeff = make_fno(mp_ea_dm, mo_energy, mo_coeff, nocc, nvir_act)
        efnoea_eae, efnoea_eav = eomeaccsd(mymf, frozen, efnoea_coeff)

        # ETO
        eto_ipe, eto_ipv = eomipccsd(mymf, frozen, ipeto_coeff)
        eto_eae, eto_eav = eomeaccsd(mymf, frozen, eaeto_coeff)

        # ENTO
        ento_ipe, ento_ipv = eomipccsd(mymf, frozen, ipento_coeff)
        ento_eae, ento_eav = eomeaccsd(mymf, frozen, eaento_coeff)

        # Canonical
        can_ipe, can_ipv = eomipccsd(mymf, frozen, mo_coeff)
        can_eae, can_eav = eomeaccsd(mymf, frozen, mo_coeff)

        # Append to lists
        data['fnoip'].append(fno_ipe)
        data['fnoea'].append(fno_eae)
        data['ntoip'].append(nto_ipe)
        data['ntoea'].append(nto_eae)
        data['enoip'].append(enoip_ipe)
        data['enoea'].append(enoea_eae)
        data['efnoip'].append(efnoip_ipe)
        data['efnoea'].append(efnoea_eae)
        data['etoip'].append(eto_ipe)
        data['etoea'].append(eto_eae)
        data['entoip'].append(ento_ipe)
        data['entoea'].append(ento_eae)
        data['canip'].append(can_ipe)
        data['canea'].append(can_eae)

    mycc = cc.RCCSD(mymf)
    eris = mycc.ao2mo()
    eccsd, t1, t2 = mycc.kernel()
    myeom = EOMIP(mycc)
    ipe, ipv = myeom.kernel(eris=eris)
    myeom = EOMEA(mycc)
    eae, eav = myeom.kernel(eris=eris)
    # Append last row for full system
    data['fnoip'].append(ipe)
    data['fnoea'].append(eae)
    data['ntoip'].append(ipe)
    data['ntoea'].append(eae)
    data['enoip'].append(ipe)
    data['enoea'].append(eae)
    data['efnoip'].append(ipe)
    data['efnoea'].append(eae)
    data['etoip'].append(ipe)
    data['etoea'].append(eae)
    data['entoip'].append(ipe)
    data['entoea'].append(eae)
    data['canip'].append(ipe)
    data['canea'].append(eae)

    # Save to file
    df = pd.DataFrame(data)
    # Save df to file
    df.to_csv('benzene_eom_data.csv', index=False)

    # plot ips
    size = berkelplot.fig_size(n_row=2, n_col=1)
    fig, ax = plt.subplots(1, 1, figsize=size)
    ax.plot(range(1, nvir+1), data['fnoip'], label='FNO')
    ax.plot(range(1, nvir+1), data['ntoip'], label='NTO')
    ax.plot(range(1, nvir+1), data['enoip'], label='ENO')
    ax.plot(range(1, nvir+1), data['efnoip'], label='EFNO')
    ax.plot(range(1, nvir+1), data['etoip'], label='ETO')
    ax.plot(range(1, nvir+1), data['entoip'], label='ENTO')
    ax.plot(range(1, nvir+1), data['canip'], label='Canonical')
    ax.set_xlabel('Number of active orbitals')
    ax.set_ylabel('IP excitation energy (Eh)')
    ax.legend()
    fig.savefig('ip_excitation_energy.png')

    # plot eas
    size = berkelplot.fig_size(n_row=2, n_col=1)
    fig, ax = plt.subplots(1, 1, figsize=size)
    ax.plot(range(1, nvir+1), data['fnoea'], label='FNO')
    ax.plot(range(1, nvir+1), data['ntoea'], label='NTO')
    ax.plot(range(1, nvir+1), data['enoea'], label='ENO')
    ax.plot(range(1, nvir+1), data['efnoea'], label='EFNO')
    ax.plot(range(1, nvir+1), data['etoea'], label='ETO')
    ax.plot(range(1, nvir+1), data['entoea'], label='ENTO')
    ax.plot(range(1, nvir+1), data['canea'], label='Canonical')
    ax.set_xlabel('Number of active orbitals')
    ax.set_ylabel('EA excitation energy (Eh)')
    ax.legend()
    fig.savefig('ea_excitation_energy.png')
    
    # Plot ip ea sum
    size = berkelplot.fig_size(n_row=2, n_col=1)
    fig, ax = plt.subplots(1, 1, figsize=size)
    ax.plot(range(1, nvir+1), np.array(data['fnoip']) + np.array(data['fnoea']), label='FNO')
    ax.plot(range(1, nvir+1), np.array(data['ntoip']) + np.array(data['ntoea']), label='NTO')
    ax.plot(range(1, nvir+1), np.array(data['enoip']) + np.array(data['enoea']), label='ENO')
    ax.plot(range(1, nvir+1), np.array(data['efnoip']) + np.array(data['efnoea']), label='EFNO')
    ax.plot(range(1, nvir+1), np.array(data['etoip']) + np.array(data['etoea']), label='ETO')
    ax.plot(range(1, nvir+1), np.array(data['entoip']) + np.array(data['entoea']), label='ENTO')
    ax.plot(range(1, nvir+1), np.array(data['canip']) + np.array(data['canea']), label='Canonical')
    ax.set_xlabel('Number of active orbitals')
    ax.set_ylabel('IP + EA excitation energy (Eh)')
    ax.legend()
    fig.savefig('ip_ea_excitation_energy.png')



