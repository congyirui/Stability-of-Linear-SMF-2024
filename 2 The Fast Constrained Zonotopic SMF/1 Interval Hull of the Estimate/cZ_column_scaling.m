function cZ_2 = cZ_column_scaling(cZ_1)
%   Scale the columns in A x = b
%   For the sake of improving the numerical precision
%
%   (c) Yirui Cong, created: 09-Oct-2021, last modified: 12-Oct-2021


G_1 = cZ_1.G;
c_1 = cZ_1.c;
A_1 = cZ_1.A;
b_1 = cZ_1.b;
cwb_1 = cZ_1.cwb;

n_g = size(A_1, 2);

G_2 = G_1;
c_2 = c_1;
A_2 = A_1;
b_2 = b_1;
cwb_2 = cwb_1;

if isempty(A_1) ~= 1
    for j = 1: n_g
        A_1_col_nonzero = A_1(:, j);
        A_1_col_nonzero = A_1_col_nonzero(A_1_col_nonzero~=0);
        a_min = min(abs(A_1_col_nonzero));
        a_max = max(abs(A_1_col_nonzero));
        a_bar = sqrt(a_min * a_max);
        
        if a_bar ~= 0
            A_2(:, j) = A_1(:, j) / a_bar;
            G_2(:, j) = G_1(:, j) / a_bar;
            cwb_2(j) = cwb_1(j) * a_bar;
        end
    end
end

cZ_2 = cZ_construct(G_2, c_2, A_2, b_2, cwb_2);