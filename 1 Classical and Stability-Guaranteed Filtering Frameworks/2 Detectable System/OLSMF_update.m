function [G_k_posterior, theta_k_posterior] = OLSMF_update(C, y_k, G_k_prior, theta_k_prior, G_v, theta_v)
%   Update step in optimal linear set-membership filtering
%   (c) Yirui Cong, created: 27-Jan-2020, last modified: 06-Feb-2020

G_k_posterior = [-G_v * C; G_k_prior];
theta_k_posterior = [theta_v - G_v * y_k; theta_k_prior];

% n = size(C, 2);

% temp = reduce_rows([G_k_posterior, theta_k_posterior]);
% G_k_posterior = temp(:,1:n);
% theta_k_posterior = temp(:,n+1);

[G_k_posterior, theta_k_posterior] = zero_row_delete(G_k_posterior, theta_k_posterior);
[G_k_posterior, theta_k_posterior] = noredund(G_k_posterior,theta_k_posterior);