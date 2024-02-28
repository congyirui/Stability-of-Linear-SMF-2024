function bd = bound_diameter_OIT(A, B, C, norm_type, d_w, d_v, delta)
%   Return an upper bound on the diameter of the OIT (new version)
%   The pair (A, C) is observable.
%
%   Inputs:
%       A - System matrix, B - Input Matrix, C - Output Matrix
%       norm_type - Indicates the type of the returned diameter: 2 - 2-norm, inf - infinity-norm
%       d_w - Upper bound of the diameter of the process noise, which is compatible with the norm_tye (e.g., when norm_type = 2, d_w is w.r.t. 2-norm)
%       d_v - Upper bound of the diameter of the measurement noise, which is compatible with the norm_tye (e.g., when norm_type = 2, d_v is w.r.t. 2-norm)
%       delta - The time-window parameter of the OIT
%
%   Output:
%       bd - Upper bound on the diameter of the OIT
%
%   (c) Yirui Cong, created: 04-Oct-2021, last modified: --


p = size(B, 2);
m = size(C, 1);

O_delta = C;
for j = 1: delta
    O_delta = [C / (A^j); O_delta];
end

matrix_bd_summand_1 = pinv(O_delta) * kron(eye(delta+1), C * pinv(C));

if norm_type == 2
    bd_summand_1 = norm(matrix_bd_summand_1, norm_type) * sqrt(delta + 1) * d_v;
elseif norm_type == inf
    bd_summand_1 = norm(matrix_bd_summand_1, norm_type) * d_v;
else
    error('Wrong type of norm!')
end

temp_sum = eye(m, p);
temp_block_diag = temp_sum;
for j = delta - 1: -1: 0
    temp_sum = temp_sum + C / (A^(delta - j)) * B;
    temp_block_diag = blkdiag(temp_sum, temp_block_diag);
end

matrix_bd_summand_2 = pinv(O_delta) * temp_block_diag;

if norm_type == 2
    bd_summand_2 = norm(matrix_bd_summand_2, norm_type) * sqrt(delta + 1) * d_w;
elseif norm_type == inf
    bd_summand_2 = norm(matrix_bd_summand_2, norm_type) * d_w;
else
    error('Wrong type of norm!')
end

bd = bd_summand_1 + bd_summand_2;