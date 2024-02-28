function flag = cZ_consistent_check(cZ, n)
%   Check if the dimensions in the input constrained zonotope are consistent
%   (c) Yirui Cong, created: 30-Apr-2021, last modified: --

flag = 0;

G = cZ.G;
c = cZ.c;
A = cZ.A;
b = cZ.b;
cwb = cZ.cwb;

if size(G, 1) ~= n
    flag = 1;
    warning('G is not correct.')
end

if size(G, 2) ~= size(A, 2)
    flag = 1;
    warning('G and A are not consistent.')
end

if size(G, 2) ~= size(cwb, 1)
    flag = 1;
    warning('G and Xi are not consistent.')
end