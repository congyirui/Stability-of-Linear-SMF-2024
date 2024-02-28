function z_out = zonotope_scaling(z_in)
%   Scale G in zonotope
%   For the sake of improving the numerical precision
%
%   (c) Yirui Cong, created: 12-Oct-2021, last modified: --

G_1 = z_in.G;
c_1 = z_in.c;
A_1 = z_in.A; % [] for zonotope
b_1 = z_in.b; % [] for zonotope
cwb_1 = z_in.cwb;

n_g = size(A_1, 2);

G_2 = G_1;
c_2 = c_1;
A_2 = A_1;
b_2 = b_1;
cwb_2 = cwb_1;

if isempty(G_1) ~= 1
    for j = 1: n_g
        G_1_col_nonzero = G_1(:, j);
        G_1_col_nonzero = G_1_col_nonzero(G_1_col_nonzero~=0);
        g_min = min(abs(G_1_col_nonzero));
        g_max = max(abs(G_1_col_nonzero));
        g_bar = sqrt(g_min * g_max);
        
        if g_bar ~= 0
            G_2(:, j) = G_1(:, j) / g_bar;
            cwb_2(j) = cwb_1(j) * g_bar;
        end
    end
end

z_out = cZ_construct(G_2, c_2, A_2, b_2, cwb_2);