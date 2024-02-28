function [A_o, B_o, C_o, flag_d, P, A_obar, A_21, B_obar, n_o] = obsv_dec(A, B, C)
%   Observability decomposition
%   Returning (A_o, B_o, C_o) in observable subsystem as well as the transformation matrix P
%   flag_d = 0 (not detectable), 1 (detectable), 2 (observable)
%   (c) Yirui Cong, created: 05-Feb-2020, last modified: 28-Dec-2020

n = size(A, 1);

[A_tilde, B_tilde, C_tilde, P_, nn] = obsvf(A, B, C);

n_o = sum(nn); % Number of states in observable subsystem

P = zeros(size(P_));
P(1: n_o, :) = P_(end-n_o+1: end, :);
P(n_o+1: end, :) = P_(1: end-n_o, :);

if n_o == 0
    error('No state is observable!')
elseif n_o < n
    A_o = A_tilde(end-n_o+1: end, end-n_o+1: end);
    B_o = B_tilde(end-n_o+1: end, :);
    C_o = C_tilde(:, end-n_o+1: end);
    
    A_obar = A_tilde(1: end-n_o, 1: end-n_o);
    A_21 = A_tilde(1: end-n_o, end-n_o+1: end);
    B_obar = B_tilde(1: end-n_o, :);
    
    if max(abs(eig(A_tilde(1: n-n_o, 1: n-n_o)))) < 1
        flag_d = 1; % Detectable
    else
        flag_d = 0; % Not detectable
    end
else
    A_o = A;
    B_o = B;
    C_o = C;
    
    A_obar = [];
    A_21 = [];
    B_obar = [];
    
    flag_d = 2; % Observable
end