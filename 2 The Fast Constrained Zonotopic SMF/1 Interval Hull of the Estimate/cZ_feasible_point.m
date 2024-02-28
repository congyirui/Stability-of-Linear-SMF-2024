function x = cZ_feasible_point(cZ)
%   Returns a feasible point in a constrained zonotope (which can be unbounded)
%   (c) Yirui Cong, created: 10-Oct-2021, last modified: 11-Feb-2022


G = cZ.G;
c = cZ.c;
A = cZ.A;
b = cZ.b;
cwb = cZ.cwb;

flag = 0;

options = optimoptions('linprog','Algorithm','dual-simplex', 'display','off');
[xi, temp_min, existflag] = linprog([], [], [], A, b, -cwb', cwb', options);

x = G * xi + c;

% if existflag ~= 1
%     error('Cannot find a feasible point!')
% end