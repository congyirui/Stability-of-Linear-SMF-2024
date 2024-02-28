function flag = cZ_point_inclusion(cZ, x)
%   Return 1 when x is in cZ, and 0 otherwise.
%   (c) Yirui Cong, created: 1-May-2021, last modified: 11-Feb-2022

G_cZ = cZ.G;
c_cZ = cZ.c;
A_cZ = cZ.A;
b_cZ = cZ.b;
cwb_cZ = cZ.cwb;

% ng = size(cZ_A, 2);

flag = 1;

options = optimoptions('linprog','Algorithm','dual-simplex', 'display','off');
[temp_x, temp_min, existflag] = linprog(ones(size(cwb_cZ')), [], [], [A_cZ; G_cZ], [b_cZ; x - c_cZ], -cwb_cZ', cwb_cZ', options);

if existflag == -2 || existflag == -5
    flag = 0;
end