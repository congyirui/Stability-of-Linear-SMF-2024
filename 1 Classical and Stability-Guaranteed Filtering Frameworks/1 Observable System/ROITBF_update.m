function [G_k_posterior, theta_k_posterior, counter] = ROITBF_update(A, C, A_o, B_o, C_o, mu_o, obsv_flag, P, k, ySequence, G_k_prior, theta_k_prior, G_w, theta_w, G_v, theta_v, counter, option)
%   Update step in ROIT-based filtering
%   option = 0 (checking if the posterior range is empty in each time step), 1 (checking at most one time for k >= \mu_o - 1)
%   counter(1) - counting the times of setting non-empty posterior for 0 <= k < \mu_o - 1
%   counter(2) - counting the times of setting non-empty posterior for k >= \mu_o - 1
%   (c) Yirui Cong, created: 06-Feb-2020, last modified: 08-Feb-2020


n = size(C, 2);
n_o = size(C_o, 2);

kIndexC = 1; % A compensator for the index 0 in matlab: for example, y[0] in matlab is y(1) = y(0 + k_indexC).
y_k = ySequence(:, k + kIndexC);


%%  Normal Update
G_k_posterior = [-G_v * C; G_k_prior];
theta_k_posterior = [theta_v - G_v * y_k; theta_k_prior];

[G_k_posterior, theta_k_posterior] = zero_row_delete(G_k_posterior, theta_k_posterior);
[G_k_posterior, theta_k_posterior] = noredund(G_k_posterior,theta_k_posterior);


%%  ROIT-Based Update
para_1 = 10;
para_S = 1;

if option == 0 || counter(2) == 0
    [x, fval, exitflag, output] = linprog(zeros(1, n), G_k_posterior, theta_k_posterior);
    if exitflag == -2 % The posterior range is empty
        if k < mu_o - 1
            counter(1) = counter(1) + 1;
            [G_ROIT, theta_ROIT] = ROIT(A_o, B_o, C_o, ySequence, k, k, G_w, theta_w, G_v, theta_v, mu_o);
            x = linprog(zeros(1, n), G_ROIT, theta_ROIT); % Find a feasible solution
            if obsv_flag <= 1
                %   Case 1: detectable but not observable
                %   Case 2: not detectable
                rr_o = size(theta_ROIT, 1);
                zero_block = zeros(rr_o, n - n_o);
                x = [x; zeros(n - n_o, 1)];
                G_k_posterior = [[G_ROIT, zero_block]; kron(eye(n), [1; -1])] / P;
                theta_k_posterior = [theta_ROIT; para_1 * ones(2*n, 1) + kron(x, [1; -1])];
                
                [G_k_posterior, theta_k_posterior] = zero_row_delete(G_k_posterior, theta_k_posterior);
                [G_k_posterior, theta_k_posterior] = noredund(G_k_posterior,theta_k_posterior);
            else
                %   Observable
                G_k_posterior = [G_ROIT; kron(eye(n), [1; -1])];
                theta_k_posterior = [theta_ROIT; para_1 * ones(2*n, 1) + kron(x, [1; -1])];
            end
        else % k >= mu_o - 1
            counter(2) = counter(2) + 1;
            [G_ROIT, theta_ROIT] = ROIT(A_o, B_o, C_o, ySequence, k, k, G_w, theta_w, G_v, theta_v, mu_o);
            if obsv_flag <= 1
                %   Case 1: detectable but not observable
                %   Case 2: not detectable
                rr_o = size(theta_ROIT, 1);
                zero_block = zeros(rr_o, n - n_o);
                temp_matrix = zeros(n - n_o, n);
                temp_matrix(:, n - n_o + 1: end) = eye(n - n_o);
                G_k_posterior = [[G_ROIT, zero_block]; kron(temp_matrix, [1; -1])] / P;
                theta_k_posterior = [theta_ROIT; para_S * ones(2*(n - n_o), 1)];
                
                [G_k_posterior, theta_k_posterior] = zero_row_delete(G_k_posterior, theta_k_posterior);
                [G_k_posterior, theta_k_posterior] = noredund(G_k_posterior,theta_k_posterior);
            else
                %   Observable
                G_k_posterior = G_ROIT;
                theta_k_posterior = theta_ROIT;
            end
        end
    end
end