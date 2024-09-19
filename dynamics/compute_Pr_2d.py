from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import warnings
warnings.simplefilter("ignore", RuntimeWarning)
# change the following to %matplotlib notebook for interactive plotting
%matplotlib inline

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(6, 6))
mpl.rc('image', cmap='gray')
plt.rcParams.update({'font.size': 22})
import numpy as np #import pandas as pd, tifffile, trackpy as tp
import numba
from numba import jit
import time

# load the coordinates
af = '65'
dia_big = 3.34
path = 'E:/old/P2-Entropy_2d/pos_binary/0.'+af
big_x = np.load(path+'/big_af-'+af+'_x.npy')*(145/512)/dia_big
big_y = np.load(path+'/big_af-'+af+'_y.npy')*(145/512)/dia_big
small_x = np.load(path+'/small_af-'+af+'_x.npy')*(145/512)/dia_big
small_y = np.load(path+'/small_af-'+af+'_y.npy')*(145/512)/dia_big

s = np.loadtxt('E:/old/P2-Entropy_2d/final calculations/softness/data/'+af+'/newL/avgphi_fij_ra=2.5_af='+af+'.txt')

na = len(big_x)
nb = len(small_x)
N = na+nb
x = np.vstack((small_x,big_x))
y = np.vstack((small_y,big_y))

print('shape allx',np.shape(x))
print('shape big',np.shape(big_x))
Nf=np.shape(x)[1]


%matplotlib notebook
plt.figure()
plt.hist(np.reciprocal(s[:,1000]),bins=60)
plt.show()

# load the relative displacements
d = 3.5
dpath = 'E:/old/P2-Entropy_2d/dmin_2d/'+af+'/sliding2100/'

# flattened

ns = 0.05
ms = 0.15
nbin = 30
ds = (ms-ns)/nbin

print('minimum S=',ns)
print('maximum S=',ms)
avg_D = 0.25
bins = np.arange(ns,ms, ds)
#print(bins)

s_all = np.array([]) 
d_all = np.array([]) 
s_high = np.array([]) 
cs = np.zeros(len(bins))
cd = np.zeros(len(bins))
psr = np.zeros(len(bins))

nall = np.array([])
nr = np.array([])
gap = 2100
Nt = 10000
for t in range(0,Nt-gap,1):  # erlier step 2
    print(t)
            
    s2 = s[:,t]
    cr = np.loadtxt(dpath+'dmin_af='+af+'_10neibs_squared_t'+str(t+1)+'-'+str(t+1+gap)+'_aveps.txt')
    D = cr[:]
    xyD = np.vstack((x[:,t].T,y[:,t],D))
    xyD = xyD.T
    xyDS = np.vstack((xyD.T,s2)).T
    xy_before = xyDS
    
    Xmin = min(xyD[:,0])
    Xmax = max(xyD[:,0])
    Ymin = min(xyD[:,1])
    Ymax = max(xyD[:,1])
        
    xyDS = xyDS[nb:N+1,:]  #xyDS[nb:N+1,:]  0:nb
    xy_before = xyDS
    insideb = np.where((xyDS[:,0]>Xmin+d)&(xyDS[:,0]<Xmax-d)&(xyDS[:,1]>Ymin+d)&(xyDS[:,1]<Ymax-d))
    xyDS = xyDS[insideb]
    xyDS[:,-1] = np.reciprocal(xyDS[:,-1])
    xy_after = xyDS
    d_all = np.append(d_all, xyDS[:,2])
    nall = np.append(nall, len(xy_after))
    idx = np.where(xyDS[:,2]>=avg_D)
    nr = np.append(nr, len(idx[0]))

    s_all = np.append(s_all, np.array(xyDS[:,3])) 
    s_high = np.append(s_high, np.array(xyDS[idx,3][0]))

for i in range(len(bins)-1):
    s1 = np.where((s_all>bins[i])&(s_all<bins[i+1]))
    cs[i] += int(np.shape(s1)[1])
    d1 = np.where((s_high>bins[i])&(s_high<bins[i+1]))
    cd[i] += int(np.shape(d1)[1])
    psr[i] += cd[i]/cs[i]
    
savedata = np.vstack((bins,(psr))).T
np.shape(savedata)

savepath = 'F:/3d_glass/review/final_data/pr/cutoff/'
np.savetxt(savepath+'fullrange_flatten_prs_DeltaT_dt=250_df=1_c=0.25_af='+af+'.txt',savedata)

plt.figure()
plt.plot(bins,np.log10(psr),'o-',label='psr 65')

plt.xlabel('$S^i$')
plt.ylabel('$P_R$')
#plt.yscale('log')
plt.legend()
plt.show()
