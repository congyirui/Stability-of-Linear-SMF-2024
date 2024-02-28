function [cZ_posterior, cZ_prior] = posterior_reset(A, B, C, cZ_prior_0, cZ_w, cZ_v, y_sequence, k, flag, B_u, u_sequence)
%   Reset the posterior (can also return the prior)
%   y_sequence contains y_0, ..., y_k which are stored in y_sequence(:, 1), ..., y_sequence(:, k+1)
%   u_sequence contains u_0, ..., u_{k-1} which are stored in u_sequence(:, 1), ..., u_sequence(:, k)
%   (c) Yirui Cong, created: 4-May-2021, last modified: 05-Oct-2022

kIndexC = 1; % A compensator for the indices in matlab: for example, y_i (0 <= i <= k) in matlab is y(:, i+1) = y(:, i + k_indexC).

%%  Initialization
temp_cZ_prior = cZ_prior_0;

%%  Recursion (Filtering Map)
for i = 0: k
    if i > 0
        %-	Prediction
        temp_cZ_prior = cZ_prediction(A, B, temp_cZ_posterior, cZ_w, B_u, u_sequence(:, i-1+kIndexC));
    end

    %-	Update
    temp_cZ_posterior = cZ_update(C, size(C, 1), y_sequence(:, i+kIndexC), temp_cZ_prior, cZ_v, flag);
end

cZ_prior = temp_cZ_prior;
cZ_posterior = temp_cZ_posterior;