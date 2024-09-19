% %  This code will calculate the MSD from relative displacement value for the 2d glass.  call this function after loading your data in run_relativeDisp_2d.
function [wmsd_eps,wmsd_aveps] = caculate_window_dmin(small_x,big_x,small_y,big_y)
af = '70';
ns = length(small_x(:,1));
nb = length(big_x(:,1));
Np=ns+nb;
savepath = strcat('/media/hdd2/softness/P2-Entropy_2d/dmin_2d/',af,'/wAvg_msd/');
filename_eps = strcat('dmin_msd_eps_big_window_averaged_af=',af,'.txt'); 
filename_aveps = strcat('dmin_msd_aveps_big_window_averaged_af=',af,'.txt');
wmsd_eps = [];
wmsd_aveps = [];
for w = 5:20:10000 % 
    fprintf('%f\n',w)
	weps = 0;
    waveps = 0;
    c=0;
    for i = 1:200:10000-w  %
		c=c+1;
        t1 = i;
		t2 = i+w;
		X1 = [small_x(:,t1)', big_x(:,t1)']';
		Y1 = [small_y(:,t1)', big_y(:,t1)']';
		
		X2 = [small_x(:,t2)', big_x(:,t2)']';
		Y2 = [small_y(:,t2)', big_y(:,t2)']';

		eps = Dmin_2D_opt_qst([X1 Y1],[X2 Y2]);
		
		averageEps = averageLocalStrain_2D_qst([X2 Y2],eps);

        xyD = [X1 Y1 eps averageEps];
        %xyD = xyD(1:ns,:);   % for small particles
        xyD = xyD(ns:Np,:);   % for big only
        %disp(size(xyD))
        %disp(ns)
        d = 3;
        
        insideb = find((xyD(:,1)>d)&(xyD(:,1)<max(xyD(:,1))-d)&(xyD(:,2)>d)&(xyD(:,2)<max(xyD(:,2))-d)); 
        xyD = xyD(insideb,:);
        %disp(size(xyD))
        weps = weps+nanmean(xyD(:,3));
        waveps = waveps+nanmean(xyD(:,4));
		
    end
   
    wmsd_eps(end+1) = weps/c;
    wmsd_aveps(end+1) = waveps/c;
    %wmsd_aveps = [wmsd_aveps waveps/c];
end 
writematrix(wmsd_eps, strcat(savepath,filename_eps))
writematrix(wmsd_aveps, strcat(savepath,filename_aveps))

end
