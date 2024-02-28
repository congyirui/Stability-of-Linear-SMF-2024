function [G_k_prior, theta_k_prior] = OLSMF_prediction(A, B, G_kminus1_posterior, theta_kminus1_posterior, G_wkminus1, theta_wkminus1)
%   Prediction step in optimal linear set-membership filtering
%   (c) Yirui Cong, created: 27-Jan-2020, last modified: 28-Jan-2020

n = size(A, 1);
% r = size(theta_kminus1_posterior, 1);
rw = size(theta_wkminus1, 1);
% p = size(B, 2);

G = [G_kminus1_posterior/A; zeros(rw, n)];
H = [-G_kminus1_posterior/A*B; G_wkminus1];
theta = [theta_kminus1_posterior; theta_wkminus1];

[G_k_prior, theta_k_prior] = sop2p(G, H, theta);