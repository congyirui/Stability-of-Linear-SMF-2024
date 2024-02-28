function cZ_posterior = OIT_inspired_estimation(A, B, C, cZ_prior_, cZ_w, cZ_v, y_sequence, k, delta, column_scaling_flag, B_u, u_sequence)
%   Returns the OIT-inspired estimate
%   y_sequence contains y_{k-delta}, ..., y_k which are stored in y_sequence(:, 1), ..., y_sequence(:, delta+1)
%   u_sequence contains u_{k-delta}, ..., u_{k-1} which are stored in u_sequence(:, 1), ..., u_sequence(:, delta)
%   (c) Yirui Cong, created: 31-Aug-2021, last modified: 07-Jan-2024

kIndexC = -k+delta+1; % A compensator for the indices in matlab: for example, y_i (k-delta <= i <= k) in matlab is y(:, i-k+delta+1) = y(:, i + k_indexC).

%%  Initialization
temp_cZ_prior = cZ_prior_;

%%  Recursion (Filtering Map)
for i = k - delta: k
    if i > k - delta
        %-	Prediction
        temp_cZ_prior = cZ_prediction(A, B, temp_cZ_posterior, cZ_w, B_u, u_sequence(:, i-1+kIndexC));
    end

    %-	Update
    temp_cZ_posterior = cZ_update(C, eye(size(C, 1)), y_sequence(:, i+kIndexC), temp_cZ_prior, cZ_v, column_scaling_flag);
%     if prior_refinement_flag == 1 && i == k - delta
%         temp_cZ_posterior = temp_cZ_prior;
%     else
%         temp_cZ_posterior = cZ_update(C, eye(size(C, 1)), y_sequence(:, i+kIndexC), temp_cZ_prior, cZ_v, column_scaling_flag);
%     end
end

cZ_posterior = temp_cZ_posterior;