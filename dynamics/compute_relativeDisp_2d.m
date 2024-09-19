%  This code will calculate the relative displacements value for the 2d glass.

function out = Compute_StrainCumulative_2D(small_x,big_x,small_y,big_y)

%tracks the particle between two subsequent stacks and calculates the strain


filename = ['d2_af=73_10neibs_squared_'];
savepath = '/media/hdd2/softness/review/final/prs2d/73/sliding5250/';

w=5250;   % frame interval used to find the deltaT = fps*tau
for i = 1 : 1 : 4750    % upto total number of frames-w
    t1 = i;
    t2 = i+w;
    X1 = [small_x(:,t1)', big_x(:,t1)']';
    Y1 = [small_y(:,t1)', big_y(:,t1)']';
    
    X2 = [small_x(:,t2)', big_x(:,t2)']';
    Y2 = [small_y(:,t2)', big_y(:,t2)']';

    eps = Dmin_2D_opt_qst([X1 Y1],[X2 Y2]);
    %fprintf('\n        ... strainfield calculated');
    tstr = ['t' int2str(t1) '-' int2str(t2)];
    fname = [filename tstr '_eps.txt']
    writematrix(eps, strcat(savepath,fname))
       
    averageEps = averageLocalStrain_2D_qst([X2 Y2],eps);
    %fprintf('\n        ... average strainfield calculated');
    fname = [filename tstr '_aveps.txt']
    writematrix(averageEps, strcat(savepath,fname))
    fprintf('\n       .... strainfield and averagestrainfield saved \n');
end

end 
