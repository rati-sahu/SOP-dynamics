function [eps] = Dmin_2D_opt_qst(a0,a1)
%calculates local strain in an amorphous system. 


% INPUT :
%       a0 - (x,y) coordinates at time t0. a1 - at later time.      


N = size(a0,1);      % number of particles
%disp(N)
NNacceptance = 1.2; %5.0     % first minimum of g(r) 
NNsq = NNacceptance*NNacceptance;
min = 0.5; 
min_sq = min*min;
D2Vec = zeros(N,2);
Dmin = zeros(N,1);

p0 = zeros(1,2);
p1 = zeros(1,2);

for ii = 1:N
    
    for j = 1 : 2
        p0(j) = a0(ii,j);     % off-atoms --------------------------
        p1(j) = a1(ii,j);
    end

    % find all the neighbors
    X0 = zeros(10,2);
    X1 = zeros(10,2);
    nn = 0;
    for j = 1 : N
           
        R = (a0(j,1)-p0(1))*(a0(j,1)-p0(1)) + ...
            (a0(j,2)-p0(2))*(a0(j,2)-p0(2));
            
        if R < NNsq && R > min_sq
            nn = nn + 1;
            X0(nn,1) = a0(j,1);
            X0(nn,2) = a0(j,2);
                
            X1(nn,1) = a1(j,1);
            X1(nn,2) = a1(j,2);
                
        end
            
    end

    DminSq = 0;       % Calculate Dmin, Dx, Dy
    for k = 1:nn
                
        dr_t0(1) = X0(k,1) - p0(1);
        dr_t0(2) = X0(k,2) - p0(2);
      
        dr_t1(1) = X1(k,1) - p1(1);
        dr_t1(2) = X1(k,2) - p1(2);
                
        for i = 1:2
            term1 = dr_t1(i)- dr_t0(i);
            term1 = term1 * term1;
            D2Vec(ii,i) = D2Vec(ii,i) + term1;
            DminSq = DminSq + term1;
        end
                
    end

            
    D2Vec(ii,1) = D2Vec(ii,1)/nn;
    D2Vec(ii,2) = D2Vec(ii,2)/nn;
            
    Dmin(ii) = DminSq/nn;
  
end 
        
eps = Dmin;
