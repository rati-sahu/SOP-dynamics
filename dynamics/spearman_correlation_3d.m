%  this code calculates the spearman rank correlation coefficient for the 3d glass and is window averaged.

dia=1.6;
path='D:/Rati/3d_glass/';
X=load(strcat(path,'STS_ASM385_70_2-12_6_T=0-24_PosX.dat'))/dia; 
Y=load(strcat(path,'STS_ASM385_70_2-12_6_T=0-24_PosY.dat'))/dia;  
Z=load(strcat(path,'STS_ASM385_70_2-12_6_T=0-24_PosZ.dat'))/dia;

[Np,Nf]=size(X);

phipath = 'D:/Rati/3d_glass/scc/2-12-6/'; 
phi = load(strcat(phipath,'phi_sts_2-12-6_t=0-24_rmin=0.82_rmax=1.5.txt'));
savepath='D:/Rati/3d_glass/scc/2-12-6/L=0/';

savedata = [];
scc_av = [];
msd_aveps = [];
for w = 3 : 2 : 25
    fprintf('%f\n',w)
    scc = 0;
    scc_array = [];
    c=0;
    msd = 0;
    datapoints = [];
    for i = 1 : 2 :Nf-w
        t1= i;
        t2 = i+w;
        eps = Dmin_3D_opt([X(:,t1) Y(:,t1) Z(:,t1)],[X(:,t2) Y(:,t2) Z(:,t2)]);
        aveps = averagelocalstrain([X(:,t2) Y(:,t2) Z(:,t2)],eps);
        
        S2 = phi(:,i);
        D = aveps(:,10); 
        xyzDS = [X(:,t1) Y(:,t1) Z(:,t1) D S2];

        d = 5;
        xmax = max(xyzDS(:,1))-d; 
        xmin = min(xyzDS(:,1))+d;  
        ymax = 25;    %max(xyzDS(:,2))-34;
        ymin = min(xyzDS(:,2))+d;
        zmax = max(xyzDS(:,3))-d; 
        zmin = min(xyzDS(:,3))+d;  
        insideb = find((xyzDS(:,1)>xmin)&(xyzDS(:,1)<xmax)&(xyzDS(:,2)>ymin)&(xyzDS(:,2)<ymax)&(xyzDS(:,3)>zmin)&(xyzDS(:,3)<zmax));
        xyzDS = xyzDS(insideb,:);

        avg_D = mean(xyzDS(:,4));
        msd = msd+avg_D;
        xyzDS(:,5) = 1./xyzDS(:,5);
        corr_coeff = corr(xyzDS(:,4), xyzDS(:,5), "type","Spearman")
        scc_array(end+1) = corr_coeff;
        %list = [xyzDS(:,5), xyzDS(:,4)];
        %datapoints = [datapoints; list];

        scc = scc + corr_coeff;
        c = c+1;
    end
    %disp(size(datapoints))
    scc_av(end+1) = scc/c;
    msd_aveps(end+1) = msd/c;
    %writematrix(datapoints, strcat(savepath,'S_dmin_2-12-6_l=0_d2_y=5-25_w=',num2str(w),'.txt'))
    avscc = mean(scc_array);
    stdscc = std(scc_array);
    each = [w avscc stdscc];
    savedata = [savedata; each];
end
writematrix(scc_av, strcat(savepath,'scc_window_2-12-6_l=0_d2_y=5-25.txt'))
size(savedata)
writematrix(savedata, strcat(savepath,'scc_window_2-12-6_l=0_d2_y=5-25_errorbar.txt'))

%writematrix(msd_aveps, strcat('dmin_rati_msd_window_avg_2-12-6_npl.txt'))
