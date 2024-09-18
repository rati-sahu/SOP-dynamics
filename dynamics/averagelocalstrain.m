function out = averagelocalstrain(X,epsilon)

% averages the strain field epsilon by averaging the epsilon tensor 
% values over the nearest neighbours of the off particle

NNacceptance = 2.2; % second minima of g(r)
NNsq = NNacceptance* NNacceptance;
N = size(X,1);
averagestrain = zeros(N,10);
for n0 = 1:N
    if epsilon(n0,1) ~= 0

        r = (X(:,1)-X(n0,1)).^2 + (X(:,2)-X(n0,2)).^2 + (X(:,3)-X(n0,3)).^2;
        neighbor = (r(:,1) < NNsq);
        if size(epsilon(neighbor),1) > 1
           averagestrain(n0,:) =  mean(epsilon(neighbor,:));
       else
           averagestrain(n0,:) = epsilon(neighbor,:);
       end
    end
end
out = averagestrain;
