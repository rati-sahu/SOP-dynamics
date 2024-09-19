#!/usr/bin/env python
# coding: utf-8

# In[63]:


import warnings
warnings.simplefilter("ignore", RuntimeWarning)
import numpy as np
import math
import matplotlib.pyplot as plt
import time
start_time = time.time()
plt.rcParams.update({'font.size': 18})
get_ipython().run_line_magic('matplotlib', 'notebook')


# load the data
fpos = 'F:/3d_glass/review/pos/'
savepath = 'F:/3d_glass/review/prs_3d_window/'
coord = 'STS_ASM385_70_2-12_4_T=0-24_'
dia = 1.6

x = np.loadtxt(fpos + coord + 'PosX.dat') / dia
y = np.loadtxt(fpos + coord + 'PosY.dat') / dia
z = np.loadtxt(fpos + coord + 'PosZ.dat') / dia


Np, Nf = x.shape

# load sop
s = np.loadtxt('E:/old/P2-Entropy/final_calculations/softness/data/2-12-4/avg_phi_sts_2-12_4_0-24_rmin=0.85_rmax=1.54_ra=2.5.txt')


gap = 18;  # step to find the D2min
Nf = 25;   # total number of frames

# range of pr bin
ns = 0.015
ms = 0.1
nbin = 30
ds = (ms-ns)/nbin
bins = np.arange(ns,ms, ds) 
alldata = np.zeros([nbin,Nf-gap])
nrp = np.zeros(Nf-gap-1)
nra = np.zeros(Nf-gap-1)

s_all = np.array([]) 
d_all = np.array([]) 

# calculate pr

for ti in range(Nf-gap-1):
    dmin = np.loadtxt('F:/3d_glass/review/prs_3d_window/dmin_d4/dmin_dt18/STS_ASM385_70_2-12_4_T=0-24_t'+str(ti)+'-'+str(ti+gap)+'_aveps.txt')

    xyzs = np.vstack((x[:,ti], y[:,ti], z[:,ti], s[:,ti])).T
    xyzs= np.vstack((xyzs.T, dmin[:,-1])).T
    #print('xyzs-shape',np.shape(xyzs))
    d = 5
    xMax = max(xyzs[:,0]) - d   
    xMin = min(xyzs[:,0]) + d
    
    yMax = max(xyzs[:,1]) - 34;     
    yMin = min(xyzs[:,1]) + d;

    zMax = max(xyzs[:,2]) - d;   
    zMin = min(xyzs[:,2]) + d;
    insideb = np.where((xyzs[:,0]>xMin)&(xyzs[:,0]<xMax)&(xyzs[:,1]>yMin)&(xyzs[:,1]<yMax)&(xyzs[:,2]>zMin)&(xyzs[:,2]<zMax))  
    
    xyzs = xyzs[insideb]
    s_in = xyzs[:,3]
    S = s_in.flatten()
    nra[ti] = len(S)
    S = np.reciprocal(S) 
    
    s_all = np.append(s_all, np.array(xyzs[:,3])) 
    d_all = np.append(d_all, np.array(xyzs[:,4])) 

    cs = np.zeros(len(bins))
    for i in range(len(S)):
        for j in range(len(bins)-1):
            if (S[i]>=bins[j]) & (S[i]<bins[j+1]):
                cs[j] = cs[j]+1
                
    hop_th = 0.04
    high = xyzs[np.where(xyzs[:,-1]>hop_th),3]
    high = high[0]
    nrp[ti] = len(high)
    high = np.reciprocal(high)
    
    cd = np.zeros(len(bins))
    for i in range(len(high)):
        for j in range(len(bins)-1):
            if (high[i]>=bins[j]) & (high[i]<bins[j+1]):
                cd[j] = cd[j]+1
                
                
    #hist_s = np.vstack((bins,cs/len(S)))
    #hist_high = np.vstack((bins,cd/len(high)))
        
    psr = np.divide(cd,cs)
    
    data = np.vstack((bins,psr)).T
    print(np.shape(data))
    alldata[:,0] = data[:,0]
    alldata[:,ti+1] = data[:,1]
    
    #plt.figure()
    #plt.plot(data[:,0],data[:,1])
    #plt.xlim(0.02,0.06)
    #plt.yscale('log')
    #plt.ylim()
    #plt.show()


mdata = np.vstack(( s_all, d_all)).T  
print(np.shape(mdata))
np.savetxt('F:/3d_glass/review/prs_3d_window/s_d2_2-12-4_dt=18_cut5.txt',mdata)


alldata2 = alldata
np.shape(alldata2)


s_values = alldata2[:,0]
alldata2 = np.delete(alldata2,0,axis=1)
alldata2 = np.delete(alldata2,-1,axis=0)
np.shape(alldata2)


means = np.mean(alldata2, axis=1)
standard_errors = np.std(np.log10(alldata2), axis=1, ddof=1) #/ np.sqrt(alldata2.shape[1])
savedata = np.vstack((s_values[0:19], means, standard_errors)).T
print(np.shape(savedata))

lnarray = np.log10(alldata2)

means = np.mean(lnarray, axis=1)
standard_errors = np.std(lnarray, axis=1, ddof=1) 

savedata_log = np.vstack((s_values[0:29], means, standard_errors)).T
print(np.shape(savedata_log))

#savedata = np.vstack((s_values[0:19], means, standard_errors)).T
np.savetxt(savepath+'Finallinear_prs_avg_std_2-12-6_dt=18f_0.01-0.12_L=2.txt',savedata)
#np.savetxt(savepath+'alldata_Finallinear_prs_avg_std_2-12-4_dt=18f_0.02-0.08.txt',alldata2)


plt.figure()

plt.errorbar(savedata[:,0], np.log10(savedata[:,1]), yerr=savedata[:,2], fmt='-', ecolor='red', capsize=5, elinewidth=1.5, markeredgewidth=1, color='k',label='rati')
#plt.errorbar(savedata_log[:,0], savedata_log[:,1], yerr=savedata_log[:,2], fmt='-', ecolor='black', capsize=5, elinewidth=1.5, markeredgewidth=1, color='k',label='log sum')
plt.xlim([0.024,0.1])
plt.ylim([-2,0])
plt.ylabel('log$P_R$')
plt.xlabel('$S$')
plt.legend()
plt.show()

