function SMF_output = OITCZ_SMF_EX(SMF_input)
%   Return the estimate of OIT-Inspired Constrained Zonotopic SMF (extended version of OIT-CZ SMF)
%
%   Inputs (Initialization - Necessary):
%       SMF_input.k - Time k
%       SMF_input.delta_bar - Parameter \bar{\delta} in Line 1 of OIT-CZ SMF
%       SMF_input.A - System Matrix
%       SMF_input.B - Input Matrix
%       SMF_input.C - Output Matrix
%       SMF_input.cZ_w - The process noise (described by a constrained zonotope) at time k-1
%       SMF_input.cZ_v - The measurement noise (described by a constrained zonotope) at time k
%       SMF_input.obsv_flag: 0 - not detectable, 1 - detectable but not observable, 2 - observable
%       SMF_input.center_translation_flag - The center translation technique: 0 - disable, 1 - enable % for extended version
%       SMF_input.prior_refinement_flag - The prior refinement technique: 0 - disable, 1 - enable % for extended version
%       SMF_input.initial_inclusion_flag: 0 - not guaranteed, 1 - guaranteed (usually used for comparing with the algorithms whose initial condition includes the true initial range) % for extended version
%       SMF_input.column_scaling_flag - Scaling the columns in A x = b of a constrained zonotope to improve the numerical precision: 0 - disable, 1 - enable % for extended version
%
%   Inputs (Initialization -  Detectable but Not Observable Systems):
%       SMF_input.A_o, SMF_input.P, SMF_input.A_obar, SMF_input.A_21, SMF_input.B_obar - Parameters in observability decomposition
%       SMF_input.epsilon, SMF_input.Upsilon_inf - in Line 1 of OIT-CZ SMF
%
%   Inputs (Filtering Process - k < \bar{\delta}):
%       SMF_input.y_sequence - Measurements y_0, ..., y_k
%       - (k = 0) SMF_input.cZ_prior_0 - Initial prior range
%       - (k > 0) SMF_input.cZ_posterior - Posterior in k-1
%   -- When the system is detectable but not observable:
%       SMF_input.cZ_prior_uos_0 - For Line 5 (Reset Posterior)
%       
%   Inputs (Filtering Process - k >= \bar{\delta}):
%       SMF_input.y_sequence - y_{k-delta}, ..., y_k
%   -- When the system is detectable but not observable:
%       SMF_input.T_hat_obar_k_minus_deltabar - \hat{\mathcal{T}}_{k-\bar{\delta}}^{\bar{o}}, used in Line 9
%       SMF_input.ell_k_minus_1 - Greatest maximum length up to k-1, used in Lines 11 and 13
%       SMF_input.c_hat_obar_k - Used in Lines 12 and 13: for k = \bar{\delta}, it should be initialized as shown in Line 12. (k > \bar{\delta})
%       SMF_input.A_obar_power - For iteratively calculating the matrix power in Line 13
%       SMF_input.d_inf_delta_bar - Used in Line 13 (k > \bar{\delta})
%   -- When SMF_input.center_translation_flag equals 1
%       SMF_input.c_hat_o - Translated center of \mathbb{R}^{n_o} % for extended version
%   -- When SMF_input.prior_refinement_flag equals 1
%       SMF_input.range_for_refinement - Interval hull of the estimate in k - \bar{\delta} (for prior refinement) % for extended version
%   
%   Outputs (Filtering Process - k < \bar{\delta})
%       SMF_output.cZ_posterior - The estimate
%   -- When the system is detectable but not observable:
%       SMF_output.T_hat_obar_k - \hat{\mathcal{T}}_k^{\bar{o}} in Line 7
%       
%   Outputs (Filtering Process - k >= \bar{\delta})
%       SMF_output.cZ_posterior - The estimate
%   -- When the system is detectable but not observable:
%       SMF_output.T_hat_obar_k - \hat{\mathcal{T}}_k^{\bar{o}} in Line 13
%       SMF_output.ell_k - Greatest maximum length up to k, in Line 11
%       SMF_output.c_hat_obar_k_plus_1 - In Line 12
%       (k = \bar{\delta}) SMF_output.d_inf_delta_bar - For Line 13
%       SMF_output.A_obar_power - For iteratively calculating the matrix power
%
%   (c) Yirui Cong, created: 03-Oct-2021, last modified: 23-Oct-2021


%%  Initialization of The Function
k = SMF_input.k; % Time k
delta_bar = SMF_input.delta_bar; % Parameter \bar{\delta}

A = SMF_input.A; % System matrix
B = SMF_input.B; % Input matrix
C = SMF_input.C; % Output matrix

n = size(A, 1);
m = size(C, 1);

cZ_w = SMF_input.cZ_w; % The process noise (described by a constrained zonotope) at time k-1
cZ_v = SMF_input.cZ_v; % The measurement noise (described by a constrained zonotope) at time k

% mu_o_upper_bound = SMF_input.mu_o_upper_bound; % An upper bound on mu_o (it equals mu for observable systems)

obsv_flag = SMF_input.obsv_flag; % obsv_flag: 0 - not detectable, 1 - detectable but not observable, 2 - observable
center_translation_flag = SMF_input.center_translation_flag; % The center translation technique: 0 - disable, 1 - enable
prior_refinement_flag = SMF_input.prior_refinement_flag; % The prior refinement technique: 0 - disable, 1 - enable
if prior_refinement_flag == 1 && delta_bar == 0
    error('The algorithm with the prior refinement is problematic when \bar{\delta} = 0; please set delta_bar > 0')
end
initial_inclusion_flag = SMF_input.initial_inclusion_flag; % initial_inclusion_flag: 0 - not guaranteed, 1 - guaranteed (usually used for comparing with the algorithms whose initial prior range includes the actual one)
column_scaling_flag = SMF_input.column_scaling_flag; % Scaling the columns in A x = b of a constrained zonotope to improve the numerical precision: 0 - disable, 1 - enable


%%  Initialization - Detectable (but not observable) Systems
if obsv_flag ~= 2
    %%  Parameters in observability decomposition
    A_o = SMF_input.A_o;
    P = SMF_input.P;
    A_obar = SMF_input.A_obar;
    A_21 = SMF_input.A_21;
    B_obar = SMF_input.B_obar;
    n_o = size(A_o, 1);
    n_obar = n - n_o;
    
    B_o = SMF_input.B_o; % For test
    C_o = SMF_input.C_o; % For test
    
    %%  Other Parameters (from Line 1)
    epsilon = SMF_input.epsilon;
    Upsilon_inf = SMF_input.Upsilon_inf;
end


%%  Lines 2-14
if k < delta_bar
    kIndexC = 1;
    
    y_sequence = SMF_input.y_sequence; % y_0, ..., y_k
    
    if k == 0
        cZ_prior = SMF_input.cZ_prior_0;
    else
        cZ_posterior = SMF_input.cZ_posterior; % Posterior in k-1
    end
    
    
    %%  Line 3: Optimal SMF
    if k > 0 && k < delta_bar
        %- Prediction Step
        cZ_prior = cZ_prediction(A, B, cZ_posterior, cZ_w);
    end

    %- Update Step
    cZ_posterior = cZ_update(C, eye(m), y_sequence(:, k+kIndexC), cZ_prior, cZ_v, column_scaling_flag);
    
    
    %%  Lines 4-6
    if is_empty_cZ(cZ_posterior) == 1
        %%  Line 5: Reset Posterior
        alpha = 1; % Initial alpha, which can be tuned
        if obsv_flag ~= 2
            cZ_prior_uos_0 = SMF_input.cZ_prior_uos_0;
            B_alpha_infinity = cZ_construct(eye(n_o), zeros(n_o, 1), [], [], alpha * ones(n_o, 1));
            temp_cZ_prior = cZ_linear_transform(inv(P), cZ_Cartesian_product(B_alpha_infinity, cZ_prior_uos_0));
        else
            B_alpha_infinity = cZ_construct(eye(n), zeros(n, 1), [], [], alpha * ones(n, 1));
            temp_cZ_prior = B_alpha_infinity;
        end

        [cZ_posterior, cZ_prior] = posterior_reset(A, B, C, temp_cZ_prior, cZ_w, cZ_v, y_sequence(:, 0+kIndexC: k+kIndexC), k, column_scaling_flag);

         while is_empty_cZ(cZ_posterior) == 1
            alpha = 2 * alpha; % Can be tuned
            if alpha > 1e150
                error('M is greater than its predefined limit!');
            end
            if obsv_flag ~= 2
                B_alpha_infinity = cZ_construct(eye(n_o), zeros(n_o, 1), [], [], alpha * ones(n_o, 1));
                temp_cZ_prior = cZ_linear_transform(inv(P), cZ_Cartesian_product(B_alpha_infinity, cZ_prior_uos_0));
            else
                B_alpha_infinity = cZ_construct(eye(n), zeros(n, 1), [], [], alpha * ones(n, 1));
                temp_cZ_prior = B_alpha_infinity;
            end

            [cZ_posterior, cZ_prior] = posterior_reset(A, B, C, temp_cZ_prior, cZ_w, cZ_v, y_sequence(:, 0+kIndexC: k+kIndexC), k, column_scaling_flag);
         end
    end
    
    
     %%  Line 7
    if obsv_flag ~= 2
        G_cZ_temp = P * cZ_prior.G;
        c_cZ_temp = P * cZ_prior.c;
        cZ_uos_prior_k = cZ_construct(G_cZ_temp(n_o+1: end, :), c_cZ_temp(n_o+1: end), cZ_prior.A, cZ_prior.b, cZ_prior.cwb);
        [G_IH_uos_prior, c_IH_uos_prior] = cZ_intervalhull(cZ_uos_prior_k);
        T_hat_obar_k = cZ_construct(eye(n_obar), c_IH_uos_prior, [], [], diag(G_IH_uos_prior));
    end
    
    
    %%  Outputs
    SMF_output.cZ_posterior = cZ_posterior;
    if obsv_flag ~= 2
        SMF_output.T_hat_obar_k = T_hat_obar_k;
    end
else
    kIndexC = -k+delta_bar+1;
    y_sequence = SMF_input.y_sequence; % y_{k-delta}, ..., y_k
    
    
    %%  Line 9: OIT-Inspired Estimate
    %-  Center Translation
    if center_translation_flag == 1
        c_hat_o = SMF_input.c_hat_o;
    else
        if obsv_flag ~= 2
            c_hat_o = zeros(n_o, 1);
        else
            c_hat_o = zeros(n, 1);
        end
    end
    
    if obsv_flag ~= 2
        T_hat_obar_k_minus_delta_bar = SMF_input.T_hat_obar_k_minus_deltabar;
        
        B_infinity_infinity = cZ_construct(eye(n_o), c_hat_o, [], [], inf(n_o, 1));
        
        %-  Prior Refinement
        if prior_refinement_flag == 1    
            if k >= 2 * delta_bar || initial_inclusion_flag == 1
                range_for_refinement = SMF_input.range_for_refinement; % It is an interval hull.
                G_range_for_refinement = P * range_for_refinement.G;
                c_range_for_refinement = P * range_for_refinement.c;
                
                temp_range_os = cZ_construct(G_range_for_refinement(1: n_o, :), c_range_for_refinement(1: n_o), [], [], range_for_refinement.cwb);
%                 temp_range_os = zonotope_scaling(temp_range_os);
                temp_cZ_prior = cZ_linear_transform(inv(P), cZ_Cartesian_product(temp_range_os, T_hat_obar_k_minus_delta_bar));
            else
                temp_cZ_prior = cZ_linear_transform(inv(P), cZ_Cartesian_product(B_infinity_infinity, T_hat_obar_k_minus_delta_bar));
            end
        else % No refinement
            temp_cZ_prior = cZ_linear_transform(inv(P), cZ_Cartesian_product(B_infinity_infinity, T_hat_obar_k_minus_delta_bar));
        end
    else
        B_infinity_infinity = cZ_construct(eye(n), c_hat_o, [], [], inf(n, 1));
        temp_cZ_prior = B_infinity_infinity;
        
        %-  Prior Refinement
        if prior_refinement_flag == 1
            if k >= 2 * delta_bar || initial_inclusion_flag == 1
                temp_cZ_prior = SMF_input.range_for_refinement; % It is an interval hull.
            end
        end
    end

    cZ_posterior = OIT_inspired_estimation(A, B, C, temp_cZ_prior, cZ_w, cZ_v, y_sequence(:, k-delta_bar+kIndexC: k+kIndexC), k, delta_bar, column_scaling_flag); % One can make it faster, since G_hat, c_hat, and A_hat are unchanged; only parts of b_hat need to be replaced.
    
    
    %%  Lines 10-13
    if obsv_flag ~= 2
        %%  Line 10: Interval Hull of the Input in Unobservable Subsystem
        G_cZ_temp = P * cZ_posterior.G;
        c_cZ_temp = P * cZ_posterior.c;
        cZ_os_posterior_k = cZ_construct(G_cZ_temp(1: n_o, :), c_cZ_temp(1: n_o), cZ_posterior.A, cZ_posterior.b, cZ_posterior.cwb);
        [G_IH_uos_input, c_IH_uos_input] = cZ_intervalhull(cZ_prediction(A_21, B_obar, cZ_os_posterior_k, cZ_w));
        
        
%         %%  Test
%         cZ_posterior_os_test = OIT_inspired_estimation(A_o, B_o, C_o, B_infinity_infinity, cZ_w, cZ_v, y_sequence(:, k-delta_bar+kIndexC: k+kIndexC), k, delta_bar, column_scaling_flag);
%         [G_IH_os, c_IH_os] = cZ_intervalhull(cZ_os_posterior_k);
%         [G_IH_os_test, c_IH_os_test] = cZ_intervalhull(cZ_posterior_os_test);
%         
%         error_G_test = norm(G_IH_os - G_IH_os_test)
%         error_G_test = norm(c_IH_os - c_IH_os_test)
%         [G_IH, c_IH] = cZ_intervalhull(cZ_posterior);
        
        
        %%  Initialization of Lines 11-13
        if k == delta_bar
            ell_k_minus_1 = 0; % For Line 11

            G_cZ_temp = P * cZ_posterior.G;
            c_cZ_temp = P * cZ_posterior.c;

            cZ_uos_posterior_k = cZ_construct(G_cZ_temp(n_o+1: end, :), c_cZ_temp(n_o+1: end), cZ_posterior.A, cZ_posterior.b, cZ_posterior.cwb);
            [G_IH_uos_posterior, c_IH_uos_posterior] = cZ_intervalhull(cZ_uos_posterior_k);
            c_hat_obar_k = c_IH_uos_posterior; % For Line 12

            d_inf_delta_bar = max(max(G_IH_uos_posterior)); % For Line 13
        else
            ell_k_minus_1 = SMF_input.ell_k_minus_1; % For Line 11
            c_hat_obar_k = SMF_input.c_hat_obar_k; % For Line 12
            d_inf_delta_bar = SMF_input.d_inf_delta_bar; % For Line 13
        end


        %%  Lines 11 and 12: \ell_k and \hat{c}_{k+1}^{\bar{o}}
        G_input_k_norm_inf = max(max(G_IH_uos_input));
        ell_k = max(G_input_k_norm_inf, ell_k_minus_1);
        
        c_hat_obar_k_plus_1 = A_obar * c_hat_obar_k + c_IH_uos_input;


        %%  Line 13: \hat{\mathcal{T}}_k^{\bar{o}}
        if k == delta_bar
            A_obar_power = eye(n_obar);
        else
            A_obar_power = SMF_input.A_obar_power * A_obar;
        end
        
        alpha_k = norm(A_obar_power, inf) * d_inf_delta_bar + Upsilon_inf * ell_k_minus_1 + epsilon;
        T_hat_obar_k = cZ_construct(eye(n_obar), c_hat_obar_k, [], [], alpha_k * ones(n_obar, 1));
%         T_hat_obar_k = cZ_construct(alpha_k * eye(n_obar), c_hat_obar_k, [], [], ones(1, n_obar));
    end
    
    
    %%  Outputs
    SMF_output.cZ_posterior = cZ_posterior;
    if obsv_flag ~= 2
        SMF_output.T_hat_obar_k = T_hat_obar_k;
        SMF_output.ell_k = ell_k;
        SMF_output.c_hat_obar_k_plus_1 = c_hat_obar_k_plus_1;
        
        if k == delta_bar
%             SMF_output.c_hat_obar_k = c_hat_obar_k;
            SMF_output.d_inf_delta_bar = d_inf_delta_bar;
        end
        
        SMF_output.A_obar_power = A_obar_power;
    end
end