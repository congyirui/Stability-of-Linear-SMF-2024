function [zono_G_IH, zono_c_IH] = cZ_OIT_inpsired_invariance_range(cZ)
%   Calculate the range for OIT-inspired invariance given a theta
%   Input: Constrained zonotope generated by filtering map
%   Output: G and C of the interval hull of the range in the subspace w.r.t. \xi_{k-\bar{\delta}}
%   (c) Yirui Cong, created: 6-Jan-2024, last modified: --

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

G = zeros(size(G_cZ));

for i = 1: n
    G(i, i) = 1;
    options = optimoptions('linprog','Algorithm','dual-simplex', 'display','off');
    [x, temp_min] = linprog(G(i, :), [], [], A_cZ, b_cZ, -cwb_cZ', cwb_cZ', options);
    [x, temp_max] = linprog(-G(i, :), [], [], A_cZ, b_cZ, -cwb_cZ', cwb_cZ', options);
    temp_max = -temp_max;
    zono_G_IH(i, i) = (temp_max - temp_min) / 2;
    zono_c_IH(i) = (temp_max + temp_min) / 2 + c_cZ(i);
end