#  This code calculates the intermediate scattering funtion ISF 
#from the cage relative displacements for a dense colloidal suspension.


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

# load data
af = '60'
dia_big=3.34 
path = '/media/hdd2/P2-Entropy_2d/pos_binary/0.'+af
big_x = np.load(path+'/big_af-'+af+'_x.npy')*(145/512)/dia_big
big_y = np.load(path+'/big_af-'+af+'_y.npy')*(145/512)/dia_big
small_x = np.load(path+'/small_af-'+af+'_x.npy')*(145/512)/dia_big
small_y = np.load(path+'/small_af-'+af+'_y.npy')*(145/512)/dia_big

# take a set of data for faster calculation
print(np.shape(big_x))
big_x = big_x[:,::100].copy()
big_y = big_y[:,::100].copy()
small_x = small_x[:,::100].copy()
small_y = small_y[:,::100].copy()
print(np.shape(big_x))

nb = len(big_x)
ns = len(small_x)
N =nb+ns
x = np.vstack((big_x,small_x))
y = np.vstack((big_y,small_y))

print(np.shape(x))

q_big =(2*np.pi)#/(dia_big)
frames=np.shape(x)[1]
j = np.sqrt(complex(-1))

# do fourier transform
def FT(x_t1, x_t2, y_t1, y_t2,q,disp_cr):
    sum1 = np.sum(np.exp(-j*q*((x_t2-x_t1)+(y_t2-y_t1)-disp_cr)))
    return sum1
  
# cage relative displacements 
# follow  https://doi.org/10.1073/pnas.1607226113  
def cr(x0,x1,y0,y1):    
	# if tag is on big then len is nb, otherwise ns
    disp_cr = np.zeros(nb)
    bulk = np.array([x0, y0])
    bulk = bulk.T
    bulk_df = pd.DataFrame(bulk[:,0:2],columns=['x','y'])
    tree = KDTree(bulk_df[['x', 'y']])
    dist_bulk, idxs = tree.query(bulk_df, k= 10, distance_upper_bound = 1.5)
    for i in range(nb):  # range is nb if tag is on big otherwise from nb to N
        dr = 0
        c = 0
        for k in range(10):
            if dist_bulk[i,k] != np.Inf and i!=k:
                dr = dr + ((x1[idxs[i,k]]-x0[idxs[i,k]])+(y1[idxs[i,k]]-y0[idxs[i,k]]))
                c = c+1
        disp_cr[i] = dr/c   # index=i if tag on big, otherwise i-nb
    return disp_cr
    
def CalcISF(tagx,tagy,allx,ally,q):
    isf = np.zeros(frames)
    for w in range(0,frames):   # w = window averaging
        lc = 0
        ft = 0
        print('w:', w)
        for ti in range(0,(frames-w)+1):
            if (ti+w<frames):
                disp_cr = cr(allx[:,ti],allx[:,ti+w],ally[:,ti],ally[:,ti+w])
                ft0 = FT(tagx[:,ti],tagx[:,ti+w],tagy[:,ti],tagy[:,ti+w],disp_cr,q) 
                re =np.real(ft0)
                im = np.imag(ft0)
                ft = ft + np.sqrt(re**2 + im**2)
                lc = lc +1
        isf[w] = ft/(len(tagx)*lc)
        print('isf[' + str(w) + ']:', isf[w])
    return isf
    
start_time = time.time()
isf_big = CalcISF(big_x, big_y,x,y,q_big)    

fps = 10/21
t = np.arange(len(isf_big))*1*fps
data = np.vstack((t,isf_big)).T
print(np.shape(data))
end_time = time.time()
print("seconds elapsed big:"+ str(end_time-start_time))
np.savetxt('/media/hdd2/P2-Entropy_2d/isf_cr/cr_final/isf_cr_10neib_rmax=5_af='+af+'_all_big_tag_dt=10.txt',data)


plt.plot(data[:,0],data[:,1],label='all')
plt.xscale('log')
plt.ylim([0,1])
plt.show()
