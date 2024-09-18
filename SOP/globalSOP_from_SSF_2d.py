#  This code AvgCagingPotential_fromSSF_2d_binary.py calculates the static structure factor(ssf) for a binary system in 2d and then the global mean field caging potential from it for a given area-fraction.
# Once you get the caging potential, the value at r=0 is the depth of the potential and by fitting the caging potential at r=0 to a harmonic form, on can get the curvature of the potential. Inverse of the depth of the potential will be the global SOP.
#  Inputs: xy coordinates
#		   Kmax : Maximum cutoff of k value. calculate upto a value where sq saturates to 1. 
#          dk : 2*pi/boxL (thermodynamic limit, you can't go below this), if you chose a very low dk then S(q) may blow up at low k.
#          Range of k: do not start from 0, for small k you will expect fluctuations due to boundary effect, so a good range will be (2*pi/boxL, Kmax, dk)

#  Outputs: S(q) for aa, ab, bb particles. aa and bb will saturate to 1, ab will saturate to 0 and ab=ba.

#  The ssf is smooth if the number of particles is high or you can average over frames.

import numpy as np
import math
import scipy.integrate as intg
from matplotlib.pyplot import *
import time

# load data
af = '65'
sigma_big = 3.34
path = '/media/hdd2/P2-Entropy_2d/pos_binary/0.'
big_x = np.loadtxt(path+af+'/big_af-'+af+'_x.dat')/sigma_big
big_y = np.loadtxt(path+af+'/big_af-'+af+'_y.dat')/sigma_big
small_x = np.loadtxt(path+af+'/small_af-'+af+'_x.dat')/sigma_big
small_y = np.loadtxt(path+af+'/small_af-'+af+'_y.dat')/sigma_big

print(max(big_x[:,0]))
Na = big_x.shape[0]
Nb = small_x.shape[0]
Nab = int(np.sqrt(Na * Nb))
N = Na+Nb
na = Na/N
nb = Nb/N

# construct kvec
boxL = 43
dk = 0.15 #2 * np.pi / boxL
kmax = dk * 150
Nk = int(np.floor(np.sqrt(2) * kmax / dk) + 1)
ky = np.arange(0.15, kmax + dk, dk)
kx = np.arange(0.15, kmax + dk, dk)
Nkx = len(kx)

Nkx2 = Nkx * Nkx
K1 = np.zeros(Nk)

kx1, ky1 = np.meshgrid(kx, ky)
kxvec = kx1.reshape(Nkx2, 1)
kyvec = ky1.reshape(Nkx2, 1)
kvec = np.hstack((kxvec, kyvec))

# algorithm for structure factor in 2d

def sq(big_x,big_y,small_x,small_y,Nt,Na,Nb,Nab,dk,Nk,Nkx2,kvec):
    
    AvgSF = np.zeros((Nk, 4))
    for t in range(Nt):
        SF = np.zeros((Nk, 4))
        print('Nt=',t)
        for kn in range(Nkx2):
            kx = kvec[kn, 0]
            ky = kvec[kn, 1]
            k = np.sqrt(kx**2 + ky**2)
            kInd = int(np.floor((k - 0.2121) / dk) + 1)
            cosa = 0
            sina = 0
            cosb = 0
            sinb = 0
            amp_big = kx*big_x[:,t] + ky*big_y[:,t]
            cosa = np.sum(np.cos(amp_big))
            sina = np.sum(np.sin(amp_big))

            amp_small = kx*small_x[:,t] + ky*small_y[:,t]
            cosb = np.sum(np.cos(amp_small))
            sinb = np.sum(np.sin(amp_small))

            SF[kInd,0] += 1  
            SF[kInd,1] += cosa*cosa + sina*sina
            SF[kInd,2] += cosa*cosb + sina*sinb 
            SF[kInd,3] += cosb*cosb + sinb*sinb 
        
        SF[:,1] = np.divide(SF[:,1],SF[:,0])
        SF[:,2] = np.divide(SF[:,2],SF[:,0])
        SF[:,3] = np.divide(SF[:,3],SF[:,0])
        
        SF[:,1] = SF[:,1]/Na
        SF[:,2] = SF[:,2]/Nab
        SF[:,3] = SF[:,3]/Nb
        
        AvgSF[:,1] += SF[:,1]
        AvgSF[:,2] += SF[:,2]
        AvgSF[:,3] += SF[:,3]
        
    return AvgSF

Nt = big_x.shape[1]
AvgSF = sq(big_x,big_y,small_x,small_y,Nt,Na,Nb,Nab,dk,Nk,Nkx2,kvec)

# Compute structure factor

start_time = time.time()
for kn in range(Nkx2):
    kx = kvec[kn,0]
    ky = kvec[kn,1]
    k = np.sqrt(kx**2 + ky**2)
    kInd = int(np.floor((k-0.2121)/dk)+1)
    K1[kInd] = k

AvgSF[:, 0] = K1
AvgSF[:, 1] = AvgSF[:, 1]/Nt
AvgSF[:, 2] = AvgSF[:, 2]/Nt
AvgSF[:, 3] = AvgSF[:, 3]/Nt
print('calculation done !')  

#np.savetxt('/media/hdd2/ssf/data/new/ssf_final_10k_af='+af+'.txt',AvgSF)
print("--- %s seconds ---" % (time.time() - start_time))

plot(AvgSF[:, 0], AvgSF[:, 1],label='aa')
plot(AvgSF[:, 0], AvgSF[:, 2],label='ab')
plot(AvgSF[:, 0], AvgSF[:, 3],label='bb')
legend()
ylabel('S(q)')
xlabel('q')
show()

dq = 0.15
q = AvgSF[int(2/dq):int(10/dq),0]    
s1 = AvgSF[int(2/dq):int(10/dq),1]
s2 = AvgSF[int(2/dq):int(10/dq),2]
s3 = AvgSF[int(2/dq):int(10/dq),3]
print(np.shape(q))
dr = 0.01
rmax = 3
r = np.arange(0, rmax, dr)
Phi_r = np.zeros(len(r))
const = (-1/(2*np.pi))

for i in range(len(r)):
	itg = np.zeros(len(q))
	for j in range(len(q)):
		det_q = s1[j]*s3[j]-s2[j]**2
		c1 = 1 - s3[j]/det_q
		c2 = s2[j]/det_q
		c3 = 1 - s1[j]/det_q
		
		#itg [j] = const * q[j] * np.exp((-(q[j]**2) * r[i]**2)/4) * (Na*c1*(s1[j]-1)+(2*c2*math.sqrt(Nb*Na)*s2[j])+c3*Nb*(s3[j]-1))
		itg [j] = const * q[j] * np.exp((-(q[j]**2) * r[i]**2)/4) * (na*c1*(s1[j]-1)+(c2*math.sqrt(na*nb)*s2[j])) # only big  
	Phi_r[i] = intg.simpson(itg,q,dq)
	
data = np.vstack((r,Phi_r)).T
#np.savetxt(savepath+'phi_r_binary_'+af+'_q=2-10.txt',data)	
plot(r,Phi_r,'*-')
show()
