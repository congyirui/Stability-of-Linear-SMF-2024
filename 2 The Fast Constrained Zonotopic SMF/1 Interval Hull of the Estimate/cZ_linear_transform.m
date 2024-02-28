function cZ_2 = cZ_linear_transform(P, cZ_1)
%   Returns cZ_2 = P * cZ_1
%   (c) Yirui Cong, created: 31-Aug-2021, last modified: --

G_cZ_2 = P * cZ_1.G;
c_cZ_2 = P * cZ_1.c;

cZ_2 = cZ_construct(G_cZ_2, c_cZ_2, cZ_1.A, cZ_1.b, cZ_1.cwb);