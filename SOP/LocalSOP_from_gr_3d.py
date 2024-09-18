# This code calculates the particle level gr for a 3d system, next the particle level caging potential and coarse-grains it over up-to the 2nd nearest neighbours.

# Inputs : linked trajectories of the particles
 
# Outputs: local gr, Caging potential of each particle at different time frame, Coarse-grained caging potential. SOP is inverse of the caging potential.


import warnings
warnings.simplefilter("ignore", RuntimeWarning)
import numpy as np
import math 
import time
import pandas as pd
import numba
from numba import jit
from scipy.spatial import cKDTree
import scipy.integrate as intg
import time

import scipy.integrate as intg
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':18})
plt.rcParams.update({'font.family': 'serif'})

##########    load data    ###########

fpos = '/media/hdd2/ShearedColloids/sr15e-6/d4/'
savepath = '/media/hdd2/P2-Entropy/final_calculations/Dmin/allTrack_t=0-20/d4/'
coord = 'STS_ASM385_70_2-12_4_T=0-19_'
dia = 1.6

x = np.loadtxt(fpos + coord + 'PosX.dat') / dia
y = np.loadtxt(fpos + coord + 'PosY.dat') / dia
z = np.loadtxt(fpos + coord + 'PosZ.dat') / dia

print('shape of x:',np.shape(x),'shape of y:',np.shape(y),'shape of z:',np.shape(z))


print('max_x',max(x[:,0]))
print('max_y',min(x[:,0]))
print('max_z',min(z[:,0]))

#################  Initialize variables   ##############

N = np.shape(x)[0]
Nf = np.shape(x)[1]
delr = 0.01
rho = N/(max(x[:,0])*max(y[:,0])*max(z[:,0]))
sigma = 0.02  
d = sigma
ds = 3*d
s = math.sqrt(2*np.pi*d*d)
maxbin = 600

###############   g(r) algorithm   ###############

@jit(nopython=True, parallel=True, nogil=True, cache=True)
def pargr(x,y,z,t):
	N = x.shape[0]
	print(N)
	histaa = np.zeros((615,N))
	aa = np.zeros((615,N))
	for j in numba.prange(N):
		for k in numba.prange(N):
			if (j != k ):
				rxjk= x[j,t]-x[k,t]
				ryjk= y[j,t]-y[k,t]
				rzjk= z[j,t]-z[k,t]
					
				rjksq = rxjk*rxjk+ryjk*ryjk+rzjk*rzjk
				rjk= math.sqrt(rjksq)
				kbin = int(rjk/delr)
				if (kbin <= maxbin-1):
					histaa[kbin,j] = histaa[kbin,j] + 1
					
##############   Gaussian brodening for continuous field   #############
	gr = np.zeros((maxbin,N+1))
	for jj in numba.prange(N):
		for ii in numba.prange(maxbin):
			rr = ii*delr
			m1 = int((rr - ds)/delr)
			l1 = int((rr + ds)/delr)
			if ( m1 >= ds ):
				for kk in numba.prange(m1,l1+1):
					ss =   kk * delr
					a11 = (rr-ss)**2

					aa[ii,jj] = aa[ii,jj] + histaa[kk,jj]*(1/s)*np.exp(-a11/(2*d*d))
			rlower = ii*delr
			rupper = rlower +delr

			xr=(rupper+rlower)/2
			ts = (1.0/(4.0*np.pi*xr*xr*rho)) 
			gr[ii,-1] = xr
			gr[ii,jj] = (ts*aa[ii,jj])
			
	return gr
	
############   calculate particle gr and bulk gr  ############

phi = np.zeros((N,Nf))
avg_phi = np.zeros((N,Nf))

bulkgr = np.zeros((600))
for f in range(1):
	rho = N/(max(x[:,f])*max(y[:,f])*max(z[:,f]))
	par_gr = pargr(x,y,z,f)
	bulkgr += np.mean(par_gr[:,0:N],axis=1)
	rall = par_gr[:,-1]
	
	############# Individual phi of Particle ###########
	
	# Start calculation form a lower rmin, from where g(r) is nonzero upto 1st minima of bulk gr.
	
	r_edges = par_gr[int(0.75/delr):int(1.3/delr),-1]    
	par_gr = par_gr[int(0.75/delr):int(1.3/delr), 0:N]
	s2 = np.zeros(N)

	for i in range(N):
		itg = np.zeros(len(r_edges))
		for j in range(len(r_edges)):
			if par_gr[j,i]!= 0:
				itg [j] = (4*np.pi*rho)*(par_gr[j,i]**2 - par_gr[j,i])*(r_edges[j]**2)
		s2[i] = intg.simpson(itg,r_edges, dx = 0.01)
	phi[:,f] = s2	
	
	plt.figure()
	plt.hist(phi[:,0],bins = 50,alpha=0.5,label='big')
	plt.xlabel('$\Phi$')
	plt.ylabel('$P(\Phi)$')
	plt.legend()
	plt.show()
	############# Average Local phi ##################

	# smooth the phi field up-to the 2nd nearest neighbours

	bulk = np.array([x[:,f], y[:,f], z[:,f]])
	bulk = bulk.T
	print(np.shape(bulk))
	bulk_df = pd.DataFrame(bulk[:,0:3],columns=['x','y','z'])
	ckdtree = cKDTree(bulk_df[['x', 'y', 'z']])
	dist_bulk, idxs = ckdtree.query(bulk_df, k= 40, distance_upper_bound= 2.2)

	av_s2 = np.zeros(N)
	ra = 2.2  
	for i in range(N):
		sum1 = 0
		sum_fij = 0
		for k in range(40):
			fij = (1-(dist_bulk[i,k]/ra)**6)/(1-(dist_bulk[i,k]/ra)**12)
			if (i != idxs[i,k]) & (idxs[i,k] != N) :
				sum1 = sum1 + s2[idxs[i,k]]*fij
				sum_fij = sum_fij + fij
		av_s2[i] = (sum1 + s2[i])/(sum_fij +1 )
	avg_phi[:,f] = av_s2
	
	plt.figure()
	plt.hist(avg_phi[:,0],bins = 50,alpha=0.5,label='big')
	plt.xlabel('$\Phi$')
	plt.ylabel('$P(\Phi)$')
	plt.legend()
	plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
