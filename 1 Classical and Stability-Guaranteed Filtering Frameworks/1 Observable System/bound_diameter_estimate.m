function bd = bound_diameter_estimate(A, B, C, norm_type, d_w, d_v)
%   Return an upper bound on the diameter of the estimate
%   If the system is observable, then the bound works for k \geq delta_optimal (corresponding to the minimum bound).
%   If the system is detectable but not observable, then the bound is in an asymptotic manner.
%
%   Inputs:
%       A - System matrix, B - Input Matrix, C - Output Matrix
%       norm_type - Indicates the type of the returned diameter: 2 - 2-norm, inf - infinity-norm
%       d_w - Upper bound of the diameter of the process noise, which is compatible with the norm_tye (e.g., when norm_type = 2, d_w is w.r.t. 2-norm)
%       d_v - Upper bound of the diameter of the measurement noise, which is compatible with the norm_tye (e.g., when norm_type = 2, d_v is w.r.t. 2-norm)
%
%   Output:
%       bd - Upper bound on the diameter of the estimate
%
%   (c) Yirui Cong, created: 04-Oct-2021, last modified: --


%%  Observability Decomposition
[A_o, B_o, C_o, obsv_flag, P, A_obar, A_21, B_obar, n_o] = obsv_dec(A, B, C);
% obsv_flag: 0 - not detectable, 1 - detectable but not observable, 2 - observable

n = size(A, 1);
n_obar = n - n_o;

r_B_o = rank(B_o);

if obsv_flag == 0
    disp('The pair (A, C) is not detectable!');
elseif obsv_flag == 1
    disp('The pair (A, C) is not observable but detectable.');
    %- Test
    test_error_max = max(max(abs(P \ [A_o zeros(n_o, n_obar); A_21 A_obar] * P - A)));
else
    disp('The pair (A, C) is observable.');
    A_o = A;
    B_o = B;
    C_o = C;
    P = eye(n);
end

mu_o = observ_index(A_o, C_o);


%%  Upper Bound
if obsv_flag ~= 2
    %- Upsilon
    rho_A_obar = max(abs(eig(A_obar)));
    num_step = 1000; % Larger num_step leads to smaller Upsilon, which can be tuned.
    numerical_step_length = (1 - rho_A_obar) / (num_step + 1);

    Upsilon = inf;
    temp_Upsilon = inf;
    for gamma = rho_A_obar + numerical_step_length: numerical_step_length: 1 - numerical_step_length
        M_gamma = 1;
        temp_M_gamma = 1;
        for temp_k = 1: 1000
            temp_M_gamma = norm((A_obar/gamma)^temp_k, norm_type);
            if temp_M_gamma > M_gamma
                M_gamma = temp_M_gamma;
            end
        end

        temp_Upsilon = M_gamma / (1 - gamma);
        if temp_Upsilon < Upsilon
            Upsilon = temp_Upsilon;
        end
    end
    
    %-  Observable subsystem
    diameter_upper_bound_os = inf;
    delta_optimal = 0;
    
    for delta_temp = mu_o - 1: 100
        diameter_upper_bound_temp = bound_diameter_OIT(A_o, B_o, C_o, norm_type, d_w, d_v, delta_temp);
        
        if diameter_upper_bound_os > diameter_upper_bound_temp
            diameter_upper_bound_os = diameter_upper_bound_temp;
            delta_optimal = delta_temp;
        end
        
        if delta_temp == mu_o - 1
            diameter_upper_bound_mu_o_minus_1 = diameter_upper_bound_temp;
        end
    end
    
    %-  Unobservable subsystem
    diameter_upper_bound_uos = Upsilon * (norm(A_21) * diameter_upper_bound_mu_o_minus_1 + norm(B_obar) * d_w);
    
    %-  The whole system
    if norm_type == 2
        bd = min(svd(P)) * sqrt(diameter_upper_bound_os^2 + diameter_upper_bound_uos^2);
    elseif norm_type == inf
        bd = norm(inv(P), inf) * max(diameter_upper_bound_os, diameter_upper_bound_uos);
    else
        error('Wrong type of norm!')
    end
else
    bd = inf;
    delta_optimal = 0;

    for delta_temp = mu_o - 1: 100
        diameter_upper_bound_temp = bound_diameter_OIT(A_o, B_o, C_o, norm_type, d_w, d_v, delta_temp);
        
        if bd > diameter_upper_bound_temp
            bd = diameter_upper_bound_temp;
            delta_optimal = delta_temp;
        end
    end
end