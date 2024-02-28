function d = estimation_gap(vertices_A, vertices_B)
%   Return the Hausdorff distance between Set A (the estimation) and Set B (the actual range).
%   Sets A and B are with V-representation (i.e., described by vertices).
%   Note that vertices_A is N_A-by-n, where N_A is the number of vertices and n isthe dimension of each vertex, so as vertices_B.
%   (c) Yirui Cong, created: 29-Sep-2021, last modified: --


N_A = size(vertices_A, 1);
N_B = size(vertices_B, 1);

d_1 = 0;

for i = 1: N_A
    d_1_inner_inf = inf;
    for j = 1: N_B
        d_temp = norm(vertices_A(i, :) - vertices_B(j, :));
        
        d_1_inner_inf = min(d_1_inner_inf, d_temp);
    end
    
    d_1 = max(d_1, d_1_inner_inf);
end

d_2 = 0;

for i = 1: N_B
    d_2_inner_inf = inf;
    for j = 1: N_A
        d_temp = norm(vertices_A(j, :) - vertices_B(i, :));
        
        d_2_inner_inf = min(d_2_inner_inf, d_temp);
    end
    
    d_2 = max(d_2, d_2_inner_inf);
end

d = max(d_1, d_2);