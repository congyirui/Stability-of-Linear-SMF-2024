function flag = is_empty_cZ(cZ)
%   Returns 1 if the constrained zonotope cZ is empty and 0 otherwise.
%   (c) Yirui Cong, created: 30-Apr-2021, last modified: 11-Feb-2022

A_cZ = cZ.A;
b_cZ = cZ.b;
cwb_cZ = cZ.cwb;

flag = 0;

options = optimoptions('linprog','Algorithm','dual-simplex', 'display','off');
[x, temp_min, existflag] = linprog(ones(size(cwb_cZ')), [], [], A_cZ, b_cZ, -cwb_cZ', cwb_cZ', options);

if existflag == -2 || existflag == -5 || existflag == -9
    flag = 1;
end

% for i = 1: ng
%     [x, temp_min, existflag] = linprog(I(i, :), [], [], cZ_A, cZ_b);
%     [x, temp_max, existflag] = linprog(-I(i, :), [], [], cZ_A, cZ_b);
%     temp_max = -temp_max;
%     %   If temp_min or temp_max is in [-1, 1], then min abs(\xi) <= 1.
%     if existflag == -3
%         continue; % unbounded
%     elseif abs(temp_min) <= 1 || abs(temp_max) <= 1
%         continue;
%     else
%         flag = 1;
%     end
% end