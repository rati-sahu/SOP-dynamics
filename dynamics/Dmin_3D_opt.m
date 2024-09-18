% function to calculate the non-affine displacements in a sheared 3d system

function [eps D2Vec] = Dmin_3D_opt(a0,a1)

%calculates local strain in an amorphous system. 


% INPUT :
%       a0 - (x,y,z) coordinates of      


N = size(a0,1);
NNacceptance = 1.3;     %first minimum of g(r) 
NNsq = NNacceptance*NNacceptance;
min = 0.6;
min_sq = min*min;
epsilontensor = zeros(N,10);
D2Vec = zeros(N,3);

p0 = zeros(1,3);
p1 = zeros(1,3);

for ii = 1:N
    
    for j = 1 : 3
        p0(j) = a0(ii,j);     % off-atoms --------------------------
        p1(j) = a1(ii,j);
    end

        FalkX = zeros(3,3);    % Initialize Tensors ---------------------
        FalkY = zeros(3,3);
        deform = zeros(3,3);

        % find all the neighbors
        X0 = zeros(20,3);
        X1 = zeros(20,3);
        N2 = size(a0,1);
        nn = 0;
        for j = 1 : N2
           
            R = (a0(j,1)-p0(1))*(a0(j,1)-p0(1)) + ...
                (a0(j,2)-p0(2))*(a0(j,2)-p0(2)) + ...
                (a0(j,3)-p0(3))*(a0(j,3)-p0(3));
            
            if R < NNsq && R > min_sq
                nn = nn + 1;
                X0(nn,1) = a0(j,1);
                X0(nn,2) = a0(j,2);
                X0(nn,3) = a0(j,3);
                
                X1(nn,1) = a1(j,1);
                X1(nn,2) = a1(j,2);
                X1(nn,3) = a1(j,3);
            end
            
        end


        if nn >=3
            for i = 1:3     % get Falk's X ------------------------------
                for j = 1:3
                    for n = 1:nn
                        FalkX(i,j) = FalkX(i,j) + (X1(n,i) - p1(i)) * (X0(n,j) - p0(j));
                    end
                end
            end
            for i = 1:3     %get Falk's Y ----------------------------------
                for j = 1:3
                    for n = 1:nn
                        FalkY(i,j) = FalkY(i,j) + (X0(n,i) - p0(i)) * (X0(n,j) - p0(j));
                    end
                end
            end
            Yinv = inv(FalkY);
            for i = 1:3       % Calculate strain -------------------------------
                for j = 1:3
                    for k = 1:3
                        deform(i,j) = deform(i,j) + FalkX(i,k) * Yinv(j,k);
                    end
                end
            end

            
            DminSq = 0;       % Calculate Dmin, Dx, Dy, Dz ---------------------
            for k = 1:nn
                
                dr_t0(1) = X0(k,1) - p0(1);
                dr_t0(2) = X0(k,2) - p0(2);
                dr_t0(3) = X0(k,3) - p0(3);
                
                dr_t1(1) = X1(k,1) - p1(1);
                dr_t1(2) = X1(k,2) - p1(2);
                dr_t1(3) = X1(k,3) - p1(3);

                for i = 1:3
                    term1 = dr_t1(i);
                    for j = 1:3
                        term1 = term1 - deform(i,j) * dr_t0(j);
                    end
                    term1 = term1 * term1;
                    D2Vec(ii,i) = D2Vec(ii,i) + term1;
                    DminSq = DminSq + term1 ;
                end
                
            end

            
            D2Vec(ii,1) = D2Vec(ii,1)/nn;
            D2Vec(ii,2) = D2Vec(ii,2)/nn;
            D2Vec(ii,3) = D2Vec(ii,3)/nn;
            DminSq = DminSq/nn;
            
            
            deform = deform - eye(3);
            deform = 1/2 * (deform + deform');
            for j = 1 : 9
                epsilontensor(ii,j) = deform(j);
            end
            epsilontensor(ii,10) = DminSq;
            
        end 
        
%    end
end
eps = epsilontensor;
