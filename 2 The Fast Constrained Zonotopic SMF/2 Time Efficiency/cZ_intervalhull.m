function [zono_G_IH, zono_c_IH] = cZ_intervalhull(cZ)
%   Interval hull of a constrained zonotope
%   Return G and c in the zonotopic description of the interval hull
%   (c) Yirui Cong, created: 4-May-2021, last modified: 11-Feb-2022

G_cZ = cZ.G;
c_cZ = cZ.c;
A_cZ = cZ.A;
b_cZ = cZ.b;
cwb_cZ = cZ.cwb;

n = size(G_cZ, 1);
% [n, ng] = size(G_cZ);
% nc = size(cZ_A, 1);

zono_G_IH = zeros(n);
zono_c_IH = zeros(n, 1);

for i = 1: n
    options = optimoptions('linprog','Algorithm','dual-simplex', 'display','off');
    [x, temp_min] = linprog(G_cZ(i, :), [], [], A_cZ, b_cZ, -cwb_cZ', cwb_cZ', options);
    [x, temp_max] = linprog(-G_cZ(i, :), [], [], A_cZ, b_cZ, -cwb_cZ', cwb_cZ', options);
    temp_max = -temp_max;
    zono_G_IH(i, i) = (temp_max - temp_min) / 2;
    zono_c_IH(i) = (temp_max + temp_min) / 2 + c_cZ(i);
end