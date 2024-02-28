function cZ_3 = cZ_intersection(cZ_1, cZ_2)
%   Returns the intersection of two constrained zonotopes (i.e., the cZ_3 is the intersection of cZ_1 and cZ_2)
%   (c) Yirui Cong, created: 03-Oct-2021, last modified: --

G_1 = cZ_1.G;
c_1 = cZ_1.c;
A_1 = cZ_1.A;
b_1 = cZ_1.b;
cwb_1 = cZ_1.cwb;

G_2 = cZ_2.G;
c_2 = cZ_2.c;
A_2 = cZ_2.A;
b_2 = cZ_2.b;
cwb_2 = cZ_2.cwb;

n = size(G_1, 1);
n_g_2 = size(G_2, 2);

G_3 = [G_1, zeros(n, n_g_2)];
c_3 = c_1;
A_3 = [blkdiag(A_1, A_2); G_1, -G_2];
b_3 = [b_1; b_2; c_2 - c_1];
cwb_3 = [cwb_1; cwb_2];

cZ_3 = cZ_construct(G_3, c_3, A_3, b_3, cwb_3);