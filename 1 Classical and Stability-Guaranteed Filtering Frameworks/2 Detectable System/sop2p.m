function [G_, theta_] = sop2p(G, H, theta, nvar)
%   Convert the description of a convex polyhedron from the form P(G, H, theta) to the form P(G_, theta_)
%   Use function fourmotz written by Sebastian Siegel

if nargin <= 3
    nvar = 1;
end

n = size(G, 2);
% l = size(H, 2);

% theta_1 = theta(1: n);
% theta_2 = theta(n+1: end);

[G_, theta_] = fourmotz([H G], theta, n);

[G_, theta_] = zero_row_delete(G_, theta_);

if nvar == 1
    [G_, theta_] = noredund(G_, theta_);
end

% r_ = size(G_, 1);
% 
% index_record_zero = [];
% 
% flag = 0;
% for i = 1: r_
%     if norm(G_(i, :)) == 0
%         index_record_zero = [index_record_zero i];
%         if theta_(i) < 0
%             warning('Infeasible!')
%         end
%         flag = 1;
%     end
% end
% 
% if flag == 1
%     G_(index_record_zero, :) = [];
%     theta_(index_record_zero, :) = [];
% end