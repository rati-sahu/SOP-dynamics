%tracks the particle between two subsequent stacks and calculates the
%strain.    

fpos = '/media/hdd2/ShearedColloids/sr15e-6/all_tracks/pos/';
savepath = '/media/hdd2/P2-Entropy/final_calculations/Dmin/allTrack_t=0-20/d6/';
coord = 'STS_ASM385_70_2-12_6_T=0-24_';
dia = 1.6;
X = load([fpos coord 'PosX.dat'])/dia;
Y = load([fpos coord 'PosY.dat'])/dia;
Z = load([fpos coord 'PosZ.dat'])/dia;
[Np, Nf] = size(X)
filename = ['STS_ASM385_70_2-12_6_T=0-24_grmin_'];

t1= 1;
for i = 3 : 2 : 24
    
    t2 = i
    eps = Dmin_3D_opt([X(:,t1) Y(:,t1) Z(:,t1)],[X(:,t2) Y(:,t2) Z(:,t2)]);  
    fprintf('\n        ... strainfield calculated');
    size(eps)
    tstr = ['t' int2str(t1-1) '-' int2str(t2-1)];
    fname = [savepath filename tstr '_eps.txt']
    save(fname,'eps','-ASCII');
    
    averageEps = averagelocalstrain([X(:,t2) Y(:,t2) Z(:,t2)],eps);   
    fprintf('\n        ... average strainfield calculated');
    fname = [savepath filename tstr '_aveps.txt']
    save(fname,'averageEps','-ASCII');
    fprintf('\n       .... strainfield and averagestrainfield saved \n');

end 
