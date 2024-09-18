# This code calculates the particle level gr for a binary system and then calculates the caging potential at a particle level. The local caging potential is coarse-grained to avoid fluctuations. local sop is reciprocal of the caging potential.
# Inputs : linked coordinates of big and small particles respectively. To make it non-dimensional we convert it to reduced units where the coordinates are divided by sigma_big(diameter of big particle).
# Outputs: gbb,gbs,gsb,gss-----> phi(u=0) and the coarse-grained caging potential. 


import numpy as np
import math 
import time
import pandas as pd
import numba
from numba import jit
from scipy.spatial import KDTree
import scipy.integrate as intg
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':18})
plt.rcParams.update({'font.family': 'serif'})
import time

sigma_big = 3.34    # diameter of particle to nondimensionalise
af = '65'
mpp = 145/512   # microns per pixel
path = 'E:/pos_binary/'

# load the coordinates
big_x = np.load(path+af+'/'+'big_af-'+af+'_x.npy')*mpp/sigma_big
big_y = np.load(path+af+'/'+'big_af-'+af+'_y.npy')*mpp/sigma_big
small_x = np.load(path+af+'/'+'small_af-'+af+'_x.npy')*mpp/sigma_big
small_y = np.load(path+af+'/'+'small_af-'+af+'_y.npy')*mpp/sigma_big

print('shape of bigx:',np.shape(big_x))
print('shape of bigy:',np.shape(big_y))
print('shape of smallx:',np.shape(small_x))
print('shape of smally:',np.shape(small_y))

print('max_big',max(big_x[:,0]))
print('min_big',min(big_x[:,0]))

# scatter plot to see the coordinates
plt.figure(figsize=(6,6))
plt.scatter(big_x[:,0],big_y[:,0],10)
plt.scatter(small_x[:,0],small_y[:,0],5)
plt.show()


Na = len(big_x)
Nb = len(small_x)
N = Na+Nb
frac_a = Na/N
frac_b = Nb/N
print('Number of big particles',Na)
print('Number of small particles',Nb)
maxbin = 300

delr = 0.01      #  should be higher than the position uncertainity 
sigma = 0.06   # vary this paramter to see if you have continuous g(r)
d = sigma
ds = 3*d
s = math.sqrt(2*np.pi*d*d)
L = max(big_y[:,0])    # box length
print('L=',L)

##################   g(r) algorithm   ################

def particle_gr(tag_x,tag_y,loop_x,loop_y,tag_n,loop_n):
	histbb = np.zeros((maxbin+50,tag_n))
	bb = np.zeros((maxbin+50,tag_n))
	for j in range(tag_n):
		for k in range(loop_n):
			if j !=k:
				rxjk= tag_x[j]-loop_x[k]
				ryjk= tag_y[j]-loop_y[k]
					
				rjksq = rxjk*rxjk+ryjk*ryjk
				rjk= math.sqrt(rjksq)
				kbin = int(rjk/delr)
				if (kbin <= maxbin-1):
					histbb[kbin,j] = histbb[kbin,j] + 1
	gr = np.zeros((maxbin,tag_n+1))
	rho = loop_n/(L*L)
	for jj in range(tag_n):
		for ii in range(maxbin):
			rr = ii*delr
			m1 = int((rr - ds)/delr)
			l1 = int((rr + ds)/delr)
			if ( m1 >= ds ):
				for kk in range(m1,l1+1):
					ss = kk * delr
					a11 = (rr-ss)**2
					bb[ii,jj] = bb[ii,jj] + histbb[kk,jj]*(1/s)*np.exp(-a11/(2*d*d))
			rlower = ii*delr
			rupper = rlower +delr

			xr=(rupper+rlower)/2
			ts = (1.0/(2.0*np.pi*xr*rho))
			gr[ii,jj] = (ts*bb[ii,jj]) 
			gr[ii,-1] = xr
	return gr

#Plot the mean local pair correlation function to get the limits of integrations, here we are doing the integtration only over the 1st peak
#In this calculation, we have used single cutoff for the g(r) as it dosen't vary much with the density of the liquid. else choose the limits according to the density.
#The variation is within +-0.1.

start_time = time.time()

frames = 1     # number of frames available 
Phi = np.zeros((N,frames))
for f in range(frames):
	print(f)
	pgr_aa = particle_gr(big_x[:,f],big_y[:,f],big_x[:,f],big_y[:,f],Na,Na)
	pgr_ab = particle_gr(big_x[:,f],big_y[:,f],small_x[:,f],small_y[:,f],Na,Nb)                  # calculate the local pair correlation function
	pgr_ba = particle_gr(small_x[:,f],small_y[:,f],big_x[:,f],big_y[:,f],Nb,Na)
	pgr_bb = particle_gr(small_x[:,f],small_y[:,f],small_x[:,f],small_y[:,f],Nb,Nb)
    
    # bigbig

	r_edges = pgr_aa[int(0.86/delr):int(1.4/delr),-1]         # give integration limits based on g(r) 1st peak
	par_gr = pgr_aa[int(0.86/delr):int(1.4/delr), 0:Na]

# 	plt.figure()
# 	r = pgr_aa[:,-1]
# 	gr_aa = np.delete(pgr_aa,-1,axis=1)          # plot the big-big g(r)
# 	gr_aa = np.mean(gr_aa,axis=1)
# 	plt.plot(r,gr_aa)
# 	plt.show()

	phi_aa = np.zeros(Na)
	rho = Na/(L*L)
	for i in range(Na):
		itg = np.zeros(len(r_edges))
		for j in range(len(r_edges)):
			if par_gr[j,i]!= 0:
				itg [j] = (2*np.pi*rho*frac_a)*(par_gr[j,i]**2 - par_gr[j,i])*(r_edges[j])
		phi_aa[i] = intg.simps(itg,r_edges, dx = 0.01)
   
    # bigsmall

	r_edges = pgr_ab[int(0.7/delr):int(1.2/delr),-1]    
	par_gr = pgr_ab[int(0.7/delr):int(1.2/delr), 0:Na]

# 	plt.figure()
# 	r = pgr_ab[:,-1]
# 	gr_ab = np.delete(pgr_ab,-1,axis=1)
# 	gr_ab = np.mean(gr_ab,axis=1)
# 	plt.plot(r,gr_ab)
# 	plt.show()

	phi_ab = np.zeros(Na)
	rho = Nb/(L*L)
	for i in range(Na):
		itg = np.zeros(len(r_edges))
		for j in range(len(r_edges)):
			if par_gr[j,i]!= 0:
				itg [j] = (2*np.pi*rho*frac_b)*(par_gr[j,i]**2 - par_gr[j,i])*(r_edges[j])
		phi_ab[i] = intg.simps(itg,r_edges, dx = 0.01)
        
	# smallbig

	r_edges = pgr_ba[int(0.7/delr):int(1.2/delr),-1]    
	par_gr = pgr_ba[int(0.7/delr):int(1.2/delr), 0:Nb]

# 	plt.figure()
# 	r = pgr_ba[:,-1]
# 	gr_ba = np.delete(pgr_ba,-1,axis=1)
# 	gr_ba = np.mean(gr_ba,axis=1)
# 	plt.plot(r,gr_ba)
# 	plt.show()
    
	phi_ba = np.zeros(Nb)
	rho = Na/(L*L)
	for i in range(Nb):
		itg = np.zeros(len(r_edges))
		for j in range(len(r_edges)):
			if par_gr[j,i]!= 0:
				itg [j] = (2*np.pi*rho*frac_a)*(par_gr[j,i]**2 - par_gr[j,i])*(r_edges[j])
		phi_ba[i] = intg.simps(itg,r_edges, dx = 0.01)
        
    # smallsmall

	r_edges = pgr_bb[int(0.57/delr):int(1.1/delr),-1]    
	par_gr = pgr_bb[int(0.57/delr):int(1.1/delr), 0:Nb]

# 	plt.figure()
# 	r = pgr_bb[:,-1]
# 	gr_bb = np.delete(pgr_bb,-1,axis=1)
# 	gr_bb = np.mean(gr_bb,axis=1)
# 	plt.plot(r,gr_bb)
# 	plt.show()

	phi_bb = np.zeros(Nb)
	rho = Nb/(L*L)
	for i in range(Nb): 
		itg = np.zeros(len(r_edges))
		for j in range(len(r_edges)):
			if par_gr[j,i]!= 0:
				itg [j] = (2*np.pi*rho*frac_b)*(par_gr[j,i]**2 - par_gr[j,i])*(r_edges[j])
		phi_bb[i] = intg.simps(itg,r_edges, dx = 0.01)
        
	phi_small = np.zeros(Nb)
	phi_big = np.zeros(Na) 
	phi_big  =  (phi_aa + phi_ab).T
	phi_small  =  (phi_bb + phi_ba).T
	phi_all = np.hstack([phi_small,phi_big])
	Phi[:,f] = phi_all
print("--- %s seconds ---" % (time.time() - start_time))

# See the distribution of bare caging potential
plt.figure()
plt.hist(phi_big,bins = 50,alpha=0.5,label='big')
plt.hist(phi_small,bins = 50,alpha=0.8,label='small')
plt.xlabel('$\Phi$')
plt.ylabel('$P(\phi)$')
plt.legend()
plt.show()

x = np.vstack((small_x, big_x))
y = np.vstack((small_y, big_y))
print(np.shape(x))


# Coarse-grain the caging potential to supress noise

av_phi = np.zeros((N,frames))

for f in range(frames):
	print(f)
	bulk = np.array([x[:,f], y[:,f]]) 
	bulk = bulk.T
# 	print(np.shape(bulk))
	bulk_df = pd.DataFrame(bulk[:,0:2],columns=['x','y'])

	tree = KDTree(bulk_df[['x', 'y']])
	dist_bulk, idxs = tree.query(bulk_df, k=40, distance_upper_bound= 2.5)    # find the neighbours within limit of 2nd neighbour

	ra = 2.5

	for i in range(N):
		sum1 = 0
		sum_fij = 0
		for k in range(40):
			if dist_bulk[i,k] != np.Inf:
				fij = (1-(dist_bulk[i,k]/ra)**6)/(1-(dist_bulk[i,k]/ra)**12)     # switch function for weight
				#if (i != idxs[i,k]) & (idxs[i,k] != Np) :
				sum1 = sum1 + Phi[idxs[i,k],f]*fij
				sum_fij = sum_fij + fij
		av_phi[i,f] = (sum1 + Phi[i,f])/(sum_fij +1)
print(np.shape(av_phi)) 

avPhi = av_phi.flatten()

# See the distribution of coarse-grained caging potential
plt.figure()
plt.hist(avPhi,bins = 50,alpha=0.8,label='avg')
#plt.hist(phi_small,bins = 50,alpha=0.8,label='small')
plt.xlabel('$\Phi$')
plt.ylabel('$P(\phi)$')
plt.legend()
plt.show()

#np.savetxt('Phi_t=0-100_af=73.txt',Phi)                            # save the caging potential values
#np.savetxt('av_Phi_ra=2.5_t=0-100_af=73.txt',av_phi)

# Remove the particles at boundaries as they do not have the correct caging potential values upto the coarse-grained length.

xyPhi = np.hstack((bulk,av_phi))
print(np.shape(xyPhi))
phib = xyPhi[Nb:N,:]  # if you want only big
print(np.shape(phib))

d=2
insideb = np.where((phib[:,0]>min(phib[:,0])+d)&(phib[:,0]<max(phib[:,0])-d)&(phib[:,1]>min(phib[:,1])+d)&(phib[:,1]<max(phib[:,1])-d))

xyPhi = phib[insideb]
print(np.shape(xyPhi))

# see the distribution after removing the edge particles

phiin = xyPhi[:,2:frames].flatten()
plt.figure()
plt.hist(np.reciprocal(phiin),bins = 100,alpha=0.8,label='big')
plt.xlabel('$S$')
plt.ylabel('$P(S)$')
plt.legend()
plt.show()

sop = xyPhi[:,2:102].flatten()
print(np.shape(sop))
sop = sop[np.where(sop>1)]
sop = np.reciprocal(sop)
print(np.shape(sop))

# plot the normalised distribution

ns = 0.1
ms = 0.2
print('ms=',ms)
print('ns=',ns)

import time
start_time = time.time()
bins = np.linspace(ns,ms,num=100)
cs = np.zeros(len(bins))
for i in range(len(sop)):
    for j in range(len(bins)-1):
        if (sop[i]>=bins[j]) & (sop[i]<bins[j+1]):
            cs[j] = cs[j]+1
            
print("--- %s seconds ---" % (time.time() - start_time))

s = np.linspace(ns,ms,num=100)
dist = np.vstack((s,cs/len(sop))).T
print(np.shape(dist))
plt.figure()
plt.plot(dist[:,0],dist[:,1],'-*')
plt.show()

#np.save('hist_sop_Af=73_old',data)

# visualise the sop field
plt.figure(figsize=(8,6))
plt.scatter(xyPhi[:,0], xyPhi[:,1],s=5, c=np.reciprocal(xyPhi[:,2]))
plt.jet() 
plt.colorbar()
plt.show()
