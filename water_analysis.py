#!/usr/bin/env python3
import pandas as pd
import berkelplot
import matplotlib.pyplot as plt
import numpy as np

# Load data from csv file
data = pd.read_csv('eom_data.csv')

# keys fnoip, ntoip, enoip, efnoip, etoip, entoip, canip
# keys fnoea, ntoea, enoea, efnoea, etoea, entoea, canea
# print

nvir = 53

# IP + EA - canip[-1] - canea[-1]
size = berkelplot.fig_size(n_row=2, n_col=1)
fig, ax = plt.subplots(1, 1, figsize=size)
ax.plot(range(1, nvir+1), data['fnoip']+data['fnoea']-data['canip'].iloc[-1]-data['canea'].iloc[-1], label='FNO')
ax.plot(range(1, nvir+1), data['ntoip']+data['ntoea']-data['canip'].iloc[-1]-data['canea'].iloc[-1], label='NTO')
ax.plot(range(1, nvir+1), data['enoip']+data['enoea']-data['canip'].iloc[-1]-data['canea'].iloc[-1], label='ENO')
ax.plot(range(1, nvir+1), data['efnoip']+data['efnoea']-data['canip'].iloc[-1]-data['canea'].iloc[-1], label='ENTO')
ax.plot(range(1, nvir+1), data['etoea']+data['etoip']-data['canip'].iloc[-1]-data['canea'].iloc[-1], label='ETO')
ax.plot(range(1, nvir+1), data['entoip']+data['entoea']-data['canip'].iloc[-1]-data['canea'].iloc[-1], label='ENTO')
ax.plot(range(1, nvir+1), data['canip']+data['canea']-data['canip'].iloc[-1]-data['canea'].iloc[-1], label='Canonical')
ax.plot((1, nvir), (0, 0), 'k--')
ax.legend()
ax.set_xlim(1, nvir)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('Number of virtual orbitals')
ax.set_ylabel('IP + EA (Hartree)')
plt.tight_layout()
plt.savefig('ip_ea.png')
