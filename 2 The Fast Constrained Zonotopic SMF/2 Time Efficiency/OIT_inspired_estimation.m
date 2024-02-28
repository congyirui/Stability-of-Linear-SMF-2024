function cZ_posterior = OIT_inspired_estimation(A, B, C, cZ_prior_, cZ_w, cZ_v, y_sequence, k, delta, flag)
%   Returns the OIT-inspired estimate
%   y_sequence contains y_{k-delta}, ..., y_k which are stored in y_sequence(:, 1), ..., y_sequence(:, delta+1)
%   (c) Yirui Cong, created: 31-Aug-2021, last modified: 09-Oct-2021

kIndexC = -k+delta+1; % A compensator for the indices in matlab: for example, y_i (k-delta <= i <= k) in matlab is y(:, i-k+delta+1) = y(:, i + k_indexC).

%%  Initialization
temp_cZ_prior = cZ_prior_;

%%  Recursion (Filtering Map)
for i = k - delta: k
    if i > k - delta
        %-	Prediction
        temp_cZ_prior = cZ_prediction(A, B, temp_cZ_posterior, cZ_w);
    end

    %-	Update
    temp_cZ_posterior = cZ_update(C, eye(size(C, 1)), y_sequence(:, i+kIndexC), temp_cZ_prior, cZ_v, flag);
end

cZ_posterior = temp_cZ_posterior;