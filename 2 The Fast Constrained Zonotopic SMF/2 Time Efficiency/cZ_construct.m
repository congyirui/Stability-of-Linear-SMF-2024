function cZ = cZ_construct(G, c, A, b, cwb)
%   Construct a constrained zonotope
%   (c) Yirui Cong, created: 4-May-2021, last modified: --

cZ.G = G;
cZ.c = c;
cZ.A = A;
cZ.b = b;
cZ.cwb = cwb; % Componentwise bound of Xi: ind(1, i) = inf means Xi^{(i)} = \mathbb{R} and ind(1, i) = 1 means Xi^{(i)} = [-1, 1].