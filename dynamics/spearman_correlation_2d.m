t = 1:10000;   % total time frames
mint = min(t);
maxt = max(t);
num_bins = 20;

log_min = log10(mint + 1);  % Avoid log(0) by adding 1
log_max = log10(maxt);
bin_edges = logspace(log_min, log_max, num_bins + 1);   % log space of time 

t = t/21;
t=t';
tau68 = 107;
tau70 = 285;
tau73 = 550;

af = '68';
path = strcat('/media/hdd2/softness/P2-Entropy_2d/pos_binary/0.',af,'/');

filename_aveps = strcat('scc_L=0_4k_big_wavg_af=',af,'.txt');
savepath = strcat('/media/hdd2/softness/');
dia=3.34;
%
big_x = load([path 'big_af-',af,'_x.dat'])/dia;
big_y = load([path 'big_af-',af,'_y.dat'])/dia;
small_x = load([path 'small_af-',af,'_x.dat'])/dia;
small_y = load([path 'small_af-',af,'_y.dat'])/dia;
%}
spath = ['/media/hdd2/softness/P2-Entropy_2d/final calculations/softness/data/',af,'/newL/'];
ss = load(strcat(spath,'phi_reducedL_sig=06_dr=002_all_t_af=',af,'.txt'));
ns = length(small_x(:,1));
nb = length(big_x(:,1));
Np=nb+ns;
scc_av = [];
for k = 1:length(bin_edges)
    w = round(bin_edges(k));
    fprintf('%f\n',w)
    scc = 0;
    c=0;
    for i = 1 : 150 : 4000-w   % final time should correspond to tau alpha
        t1 = i;
        s = ss(:,t1);
        t2 = t1+w;
        X1 = [small_x(:,t1)', big_x(:,t1)']';
        Y1 = [small_y(:,t1)', big_y(:,t1)']';
        
        X2 = [small_x(:,t2)', big_x(:,t2)']';
        Y2 = [small_y(:,t2)', big_y(:,t2)']';
    
        eps = Dmin_2D_opt_qst([X1 Y1],[X2 Y2]);
        averageEps = averageLocalStrain_2D_qst([X2 Y2],eps);
        
        xysD = [X1 Y1 averageEps s];
        xysD = xysD(ns:Np,:);
        d = 4;
        insideb = find((xysD(:,1)>d)&(xysD(:,1)<max(xysD(:,1))-d)&(xysD(:,2)>d)&(xysD(:,2)<max(xysD(:,2))-d)); 
        xysD = xysD(insideb,:);
        xysD(:,4) = 1./xysD(:,4);
        if ~isnan(xysD(:,3))
            scc = scc + corr(xysD(:,3), xysD(:,4), "type","Spearman");
            c=c+1;
        end
    end
    scc_av(end+1) = scc/c;
end 

writematrix(scc_av, strcat(savepath,filename_aveps))
plot(bin_edges/(21*107),scc_av,'*-')
xlabel('$t/\tau_{\alpha}$','FontSize',50,'interpreter','latex')
ylabel('$SCC(S^i,d_i^2)$','FontSize',20,'interpreter','latex')
set(gca,'FontSize',28);
set(gca,'XScale','log');
ylim([0,0.8])
