function cZ_3 = cZ_Cartesian_product(cZ_1, cZ_2)
%   Returns the Cartesian product of two constrained zonotopes
%   (c) Yirui Cong, created: 30-Apr-2021, last modified: 3-May-2021

G_cZ_3 = blkdiag(cZ_1.G, cZ_2.G);
c_cZ_3 = [cZ_1.c; cZ_2.c];

ng_1 = size(cZ_1.G, 2);
nc_1 = size(cZ_1.A, 1);
ng_2 = size(cZ_2.G, 2);
nc_2 = size(cZ_2.A, 1);

if isempty(cZ_1.A) && isempty(cZ_2.A)
    A_cZ_3 = [];
    b_cZ_3 = [];
elseif isempty(cZ_1.A)
    A_cZ_3 = [zeros(nc_2, ng_1), cZ_2.A];
    b_cZ_3 = cZ_2.b;
elseif isempty(cZ_2.A)
    A_cZ_3 = [cZ_1.A, zeros(nc_1, ng_2)];
    b_cZ_3 = cZ_1.b;
else
    A_cZ_3 = [cZ_1.A, zeros(nc_1, ng_2); zeros(nc_2, ng_1), cZ_2.A];
    b_cZ_3 = [cZ_1.b; cZ_2.b];
end

cwb_cZ_3 = [cZ_1.cwb; cZ_2.cwb];

% A_cZ_3 = blkdiag(cZ_1.A, cZ_2.A);
% b_cZ_3 = [cZ_1.b; cZ_2.b];

cZ_3 = cZ_construct(G_cZ_3, c_cZ_3, A_cZ_3, b_cZ_3, cwb_cZ_3);