#  This code globalSOP_from_ssf_3d.py calculates the static structure factor(ssf) for a monodisperse 3d system(For systems with less than 7% polydispersity, this code can be used). From the ssf the global Mean field caging potential is calculated.
# Once you get the caging potential, the value at r=0 is the depth of the potential and by fitting the caging potential at r=0 to a harmonic form, on can get the curvature of the potential. Inverse of the depth of the potential will be the global SOP.

#  Inputs: xyz coordinates
#		   Kmax : Does not matter how large value you take. calculate upto a value where sq saturates to 1.
#          dk : 2pi/boxl here
#          Range of k: do not include 0, for small k you will expect fluctuations, this is due to boundary effects, so a good range will be (2*pi/boxL, Kmax, dk)

#  Outputs: S(q) for particles. Should saturate to 1.

#  The ssf is smooth if the number of particles is high or you can average over frames.

import numpy as np
import scipy.integrate as intg
from matplotlib.pyplot import *
import time


#   algorithm for structure factor
def sq(x, y, z, N, dk, Nk, Nkx3 ,kvec, kmin):

    K1 = np.zeros(Nk)
    SF = np.zeros((Nk, 2))
    for kn in range(Nkx3):
        kx = kvec[kn, 0]
        ky = kvec[kn, 1]
        kz = kvec[kn, 2]

        k = np.sqrt(kx**2 + ky**2 + kz**2)
        kInd = int(np.floor((k-kmin) / dk) + 1)
        K1[kInd] = k

        cosa = 0
        sina = 0

        amp = kx*x + ky*y + kz*z
        cosa = np.sum(np.cos(amp))
        sina = np.sum(np.sin(amp))

        SF[kInd,0] += 1
        SF[kInd,1] += cosa*cosa + sina*sina
    
    SF[:,1] = np.divide(SF[:,1],SF[:,0])
    
    SF[:,1] = SF[:,1]/N
    
    return SF[:,1], K1

# calculate ssf

start_time = time.time()
boxl = 60 #max(z)  # box length
dk = 0.1#(2 * np.pi) /boxl
kmax = dk * 100
Nk = int((3 * kmax / dk) + 1)
ky = np.arange(dk, kmax + dk, dk)
kx = np.arange(dk, kmax + dk, dk)
kz = np.arange(dk, kmax + dk, dk)

Nkx = len(kx)

Nkx3 = Nkx * Nkx * Nkx
kmin = np.sqrt(3*(dk**2))

kx1, ky1, kz1 = np.meshgrid(kx, ky, kz)
kxvec = kx1.reshape(Nkx3, 1)
kyvec = ky1.reshape(Nkx3, 1)
kzvec = kz1.reshape(Nkx3, 1)
kvec = np.hstack((kxvec, kyvec, kzvec))
AvgSF = np.zeros(Nk)

dia = 1.3
for t in range(0,1):
	print(t)
	if t<10:
		data = np.loadtxt('/media/hdd2/ShearedColloids/sr15e-6/features_kilfoil/d4/STS_ASM385_70_2-12_4_0'+str(t)+'.txt')
	else:
		data = np.loadtxt('/media/hdd2/ShearedColloids/sr15e-6/features_kilfoil/d4/STS_ASM385_70_2-12_4_'+str(t)+'.txt')
	x =  data[:,0]*0.206/dia
	y =  data[:,1]*0.206/dia         # multiply the micron per pixel here
	z =  data[:,2]*0.15/dia
	
	d = 25   # if you want to do it over a small box, else set to 0
	
	xyz = np.vstack((x[:],y[:],z[:])).T
	idx= np.where((xyz[:,0]>d)&(xyz[:,0]<max(xyz[:,0])-d)&(xyz[:,1]>d)&(xyz[:,1]<max(xyz[:,1])-d)&(xyz[:,2]>d)&(xyz[:,2]<max(xyz[:,2])-d))
	xyz_in = xyz[idx]
	print(np.shape(xyz_in))
	N = xyz_in.shape[0]
	SF, K = sq(xyz_in[:,0], xyz_in[:,1], xyz_in[:,2], N, dk, Nk, Nkx3, kvec, kmin)
	ssf = np.vstack((K,SF)).T

plot(ssf[:,0],ssf[:,1],'*-')
show()
# Calculate mean field caging potential, average the ssf over several frames to remove noise. Integration is done over the 1st peak of the ssf. 

k = ssf[int(4/dk):int(10/dk), 0]    
sq = ssf[int(4/dk):int(10/dk), 1]
dr = 0.05
rmax = 1.5
r = np.arange(0, rmax, dr)
Phi_r = np.zeros(len(r))
const = (-1/(2*np.pi**2))
for i in range(len(r)):
	itg = np.zeros(len(sq))
	for j in range(len(sq)):
		itg [j] = const * (k[j]**2) * (sq[j]-1)**2 * np.exp((-(k[j]**2) * r[i]**2)/6) 
	Phi_r[i] = intg.simpson(itg,k, dk)

plot(r,Phi_r,'*-')
show()
