%%% Example B - 1 - Detectable Systems
%   Randomly selected initial conditions
%   p = m = n_o
%
%   (c) Yirui Cong, created: 08-Oct-2021, last modified: 17-Feb-2024

close all;
clear;
clc;

rng(1); % For reproducible results


%%  Simulation Parameters
k_max = 100;
k_sequence = 0: k_max;
kIndexC = 1; % A compensator for the index 0 in matlab: for example, y_0 in matlab is y(:,1) = y(0 + k_indexC).

n = 10;
set_of_n_o = 7: 9; % The range of n_o (note that m = p = n_o in this simulation)

num_set_of_n_obar = length(set_of_n_o);
num_trials_each = 1000; % Number of trials for each p


%%  Data Storage - Multiple Trials
inclusion_indicators = ones(k_max + kIndexC, num_trials_each, num_set_of_n_obar); % Indicators of if the state is included in the estimate
diameters = zeros(k_max + kIndexC, num_trials_each, num_set_of_n_obar); % Diameter of interval hull
diameters_inf = zeros(k_max + kIndexC, num_trials_each, num_set_of_n_obar); % Diameter of the estimate, in the sense of infinity-norm
volumes = zeros(k_max + kIndexC, num_trials_each, num_set_of_n_obar); % Volume of interal hull
bounds_estimation_gap = zeros(k_max + kIndexC, num_trials_each, num_set_of_n_obar); % Bound on estimation gap
bounds_estimation_gap_inf = zeros(k_max + kIndexC, num_trials_each, num_set_of_n_obar); % Bound on estimation gap, in the sense of infinity-norm


%%  Highlighted Trials
num_highlights = 100;
highlights = cell(num_highlights, num_set_of_n_obar);


%%  Trials
counter_n_o = 0;
tic
for n_o = set_of_n_o
    p = n_o;
    m = n_o;
    
    n_obar = n - n_o;
    
    counter_n_o = counter_n_o + 1;
    
    for n_trial = 1: num_trials_each
        n_o
        n_trial
        
        %%  System Parameters
        flag_successfully_generated = 0;
        
        while flag_successfully_generated == 0
            %-  Observable Subsystem
            sys = drss(n_o, m, p);

            A_o = sys.A;
            B_o = sys.B;
            C_o = sys.C;
            
            %-  Unobservable Subsystem
            matrix_temp_1 = rand(n_obar);
            scale_temp = rand * 0.5; % Determines the "overshoot"
            spectral_radius = max(abs(eig(matrix_temp_1)));
            A_obar = matrix_temp_1 / spectral_radius * scale_temp;
            
            A_21 = rand(n_obar, n_o);
            B_obar = rand(n_obar, p);
            
            %-  The Whole System
            matrix_temp_2 = rand(n);
            [Q_temp, R_temp] = qr(matrix_temp_2);
            P = Q_temp;
            
            A = P \ [A_o zeros(n_o, n_obar); A_21 A_obar] * P;
            B = P \ [B_o; B_obar];
            C = [C_o zeros(m, n_obar)] * P;
            
            if rank(obsv(A_o, C_o)) == n_o
                flag_successfully_generated = 1;
            else
                continue; % The numerical precision might not be high enough to support a bounded estimate.
            end
            
            obsv_flag = 1;
        end

        %-  Process Noise
        %-- Interval-Type Noise
        cZ_w = cZ_construct([], [], [], [], []);
        cZ_w.G = eye(p);
        cZ_w.c = zeros(p, 1);
        cZ_w.A = [];
        cZ_w.b = [];
        cZ_w.cwb = ones(p, 1);

        %-  Measurement Noise
        %-- Interval-Type Noise
        cZ_v = cZ_construct([], [], [], [], []);
        cZ_v.G = eye(m);
        cZ_v.c = zeros(m, 1);
        cZ_v.A = [];
        cZ_v.b = [];
        cZ_v.cwb = ones(m, 1);

        %-  True Initial Range
        cZ_G_0_real = 10 * eye(n);
        cZ_c_0_real = zeros(n, 1);

        % cZ_G_0_real = 10 * eye(n);
        % cZ_c_0_real = zeros(n, 1) + 100;


       %%  Initialization of OIT-CZ SMF (Line 1 in Algorithm 3)
        k = 0;

        G_cZ_0_prior_OITCZSMF = cZ_G_0_real;
        c_cZ_0_prior_OITCZSMF = cZ_c_0_real + 10 * 2 * (rand(n, 1) - 0.5);
        % cZ_c_0_prior_OITCZSMF = cZ_c_0_real + 10;
        A_cZ_0_prior_OITCZSMF = [];
        b_cZ_0_prior_OITCZSMF = [];
        cwb_cZ_0_prior_OITCZSMF = ones(size(cZ_G_0_real, 2), 1);
        cZ_prior_OITCZSMF_0 = cZ_construct(G_cZ_0_prior_OITCZSMF, c_cZ_0_prior_OITCZSMF, A_cZ_0_prior_OITCZSMF, b_cZ_0_prior_OITCZSMF, cwb_cZ_0_prior_OITCZSMF);

        r_C_o = rank(C_o);
        delta_bar = (n_o-r_C_o)+3;
        if obsv_flag ~= 2
            epsilon = 1e-3;
        end

        %- Upsilon_{\infty}
        if obsv_flag ~= 2
            rho_A_obar = max(abs(eig(A_obar)));
            num_step = 1000; % Larger num_step leads to smaller Upsilon, which can be tuned.
            numerical_step_length = (1 - rho_A_obar) / (num_step + 1);

            Upsilon_inf = inf;
            temp_Upsilon = inf;
            for gamma = rho_A_obar + numerical_step_length: numerical_step_length: 1 - numerical_step_length
                M_gamma = 1;
                temp_M_gamma = 1;
                for temp_k = 1: 1000
                    temp_M_gamma = norm((A_obar/gamma)^temp_k, inf);
                    if temp_M_gamma > M_gamma
                        M_gamma = temp_M_gamma;
                    end
                end

                temp_Upsilon = M_gamma / (1 - gamma);
                if temp_Upsilon < Upsilon_inf
                    Upsilon_inf = temp_Upsilon;
                end
            end
        end

        % if delta_o < mu_o_upperbound - 1
        %     error('delta_o is smaller than mu_o - 1!')
        % end
        %%----------------------------------------------------------

        %- Unobservale part of initial condition (used in Line 5)
        if obsv_flag ~= 2
            G_cZ_temp = P * G_cZ_0_prior_OITCZSMF;
            c_cZ_temp = P * c_cZ_0_prior_OITCZSMF;
            cZ_prior_uos_OITCZSMF_0 = cZ_construct(G_cZ_temp(n_o+1: end, :), c_cZ_temp(n_o+1: end), A_cZ_0_prior_OITCZSMF, b_cZ_0_prior_OITCZSMF, cwb_cZ_0_prior_OITCZSMF);
            
%             cZ_prior_uos_OITCZSMF_0 = zonotope_scaling(cZ_prior_uos_OITCZSMF_0);
        end

        % %- For iteratively calculating the matrix power in Line 13
        % if obsv_flag ~= 2
        %     A_obar_power = eye(n_obar);
        % end


        %%  Initialization of OITCZ_SMF Function
        OITCZ_SMF_input.delta_bar = delta_bar;
        OITCZ_SMF_input.A = A;
        OITCZ_SMF_input.B = B;
        OITCZ_SMF_input.C = C;
        OITCZ_SMF_input.cZ_w = cZ_w;
        OITCZ_SMF_input.cZ_v = cZ_v;
        OITCZ_SMF_input.obsv_flag = obsv_flag;

%         OITCZ_SMF_input.mu_o_upper_bound = n_o-r_C_o+1;

        OITCZ_SMF_input.center_translation_flag = 0;
        OITCZ_SMF_input.prior_refinement_flag = 1;
        OITCZ_SMF_input.initial_inclusion_flag = 0;
        
        OITCZ_SMF_input.column_scaling_flag = 0;

        if obsv_flag ~= 2
            OITCZ_SMF_input.A_o = A_o;
            OITCZ_SMF_input.P = P;
            OITCZ_SMF_input.A_obar = A_obar;
            OITCZ_SMF_input.A_21 = A_21;
            OITCZ_SMF_input.B_obar = B_obar;
            
            OITCZ_SMF_input.B_o = B_o; % For invariance test
            OITCZ_SMF_input.C_o = C_o; % For invariance test

            OITCZ_SMF_input.epsilon = epsilon;
            OITCZ_SMF_input.Upsilon_inf = Upsilon_inf;
        end


        %%  Data Storage
        cZ_posterior_OITCZSMF = cell(k_max+kIndexC, 1); % Posterior
        IH_posterior_OITCZSMF = cell(k_max+kIndexC, 1); % Interval hull of posterior
        IH_posterior_OITCZSMF_real = cell(k_max+kIndexC, 1); % Interval hull of posterior (the center is translated back)

        if obsv_flag ~= 2
            T_hat_obar = cell(k_max+kIndexC, 1); % Store \hat{\mathcal{T}}_k^{\bar{o}}
            ell = zeros(1, k_max+kIndexC);
            c_hat_obar = zeros(n_obar, k_max+kIndexC+1);
        end

        x_sequence = zeros(n, k_max+kIndexC);
        y_sequence = zeros(m, k_max+kIndexC);
        w_sequence = cZ_w.G * 2 * (rand(p, k_max+kIndexC) - 0.5) + cZ_w.c; % Process noises
        v_sequence = cZ_v.G * 2 * (rand(m, k_max+kIndexC) - 0.5) + cZ_v.c; % Measurement noises
        
        
        %%  Numerical Stability of Floating-Point Calculations
        %-- For [1e16-1, 1e16+1], the floating numbers cannot correctly record this range due to the precision limit. This usually happens for unstable systems,
        %   which can result in unexpected stops in SMFing algorithms. To solve this problem, we introduce a virtual control input-based method.
        
        u_sequence = zeros(n, k_max+kIndexC); % Virtual control input (for the numerical stability of floating-point calculations when x is large)
        B_u = eye(n); % The input matrix w.r.t. the virtual control input
        OITCZ_SMF_input.B_u = B_u;
        
        B_u_o = P(1: n_o, :) * B_u;
        B_u_obar = P(n_o+1: end, :) * B_u;
        OITCZ_SMF_input.B_u_obar = B_u_obar;
        
        component_threshold = 1e4; % Whenever a component of x_k exceeds component_threshold, the virtual control input will translate the state close to the origin to keep a good precision.
        
        x_sequence_difference = zeros(n, k_max+kIndexC); % It records the difference between the real state and the translated state.
        
        
        %%  Highlighted Parts
        if n_trial <= num_highlights
            highlights{n_trial, counter_n_o}.A = A;
            highlights{n_trial, counter_n_o}.B = B;
            highlights{n_trial, counter_n_o}.C = C;
            
            highlights{n_trial, counter_n_o}.x_sequence = zeros(n, k_max+kIndexC);
            highlights{n_trial, counter_n_o}.y_sequence = zeros(m, k_max+kIndexC);
            highlights{n_trial, counter_n_o}.cZ_posterior_OITCZSMF = cell(k_max+kIndexC, 1);
            highlights{n_trial, counter_n_o}.IH_posterior_OITCZSMF = cell(k_max+kIndexC, 1);
            highlights{n_trial, counter_n_o}.IH_posterior_OITCZSMF_real = cell(k_max+kIndexC, 1);
            
            highlights{n_trial, counter_n_o}.k_reset = zeros(size(k_sequence));
        end


        %%  Simulations
        for k = k_sequence
            k
            OITCZ_SMF_input.k = k;


            %%  Realizations of States and Measurements
            if k == 0
                x_sequence(:, kIndexC) = cZ_G_0_real * 2 * (rand(n, 1) - 0.5) + cZ_c_0_real;
            else
                x_sequence(:, k+kIndexC) = A * x_sequence(:, k-1+kIndexC) + B * w_sequence(:, k-1+kIndexC);
                
                if max(abs(x_sequence(:, k+kIndexC))) > component_threshold
                    u_sequence(:, k-1+kIndexC) = - A * zono_c_IH_OITCZSMF - B * cZ_w.c;
%                     u_sequence(:, k-1+kIndexC) = - A * cZ_posterior_OITCZSMF{k-1+kIndexC}.c - B * cZ_w.c;
%                     u_sequence(:, k-1+kIndexC) = - x_sequence(:, k+kIndexC);
                    x_sequence(:, k+kIndexC) = x_sequence(:, k+kIndexC) + B_u * u_sequence(:, k-1+kIndexC);
                    
                    if obsv_flag ~= 2
                        if k >= delta_bar
                            c_hat_obar(:, k+kIndexC) = c_hat_obar(:, k+kIndexC) + B_u_obar * u_sequence(:, k-1+kIndexC); % Used in Line 11 (Compensation in Line 9 -- the original code does not take the input into account since the virtual control input was not derived)
                        end
                    end
                end
                
                x_sequence_difference(:, k+kIndexC) = A * x_sequence_difference(:, k-1+kIndexC) - u_sequence(:, k-1+kIndexC);
            end

            y_sequence(:, k+kIndexC) = C * x_sequence(:, k+kIndexC) + v_sequence(:, k+kIndexC);


            %%  OIT-CZ SMF
            if k < delta_bar
                OITCZ_SMF_input.y_sequence = y_sequence(:, kIndexC: k+kIndexC);
                OITCZ_SMF_input.u_sequence = u_sequence(:, kIndexC: k-1+kIndexC);
                if k == 0
                    OITCZ_SMF_input.cZ_prior_0 = cZ_prior_OITCZSMF_0; % Initial condition
                else
                    OITCZ_SMF_input.cZ_posterior = cZ_posterior_OITCZSMF{k-1+kIndexC}; % Posterior in k-1
                end

                if obsv_flag ~= 2
                    OITCZ_SMF_input.cZ_prior_uos_0 = cZ_prior_uos_OITCZSMF_0;
                end

                OITCZ_SMF_output = OITCZ_SMF_EX(OITCZ_SMF_input);

                cZ_posterior_OITCZSMF{k+kIndexC} = OITCZ_SMF_output.cZ_posterior; % The estimate
                if obsv_flag ~= 2
                    T_hat_obar{k+kIndexC} = OITCZ_SMF_output.T_hat_obar_k; % \hat{\mathcal{T}}_k^{\bar{o}} in Line 7 of Algorithm 3
                end
                
                %-  Highlighted Parts
                if n_trial <= num_highlights
                    highlights{n_trial, counter_n_o}.k_reset(k+kIndexC) = OITCZ_SMF_output.flag_reset;
                end
            else
                OITCZ_SMF_input.y_sequence = y_sequence(:, k-delta_bar+kIndexC: k+kIndexC); % y_{k-delta}, ..., y_k
                OITCZ_SMF_input.u_sequence = u_sequence(:, k-delta_bar+kIndexC: k-1+kIndexC);
                if obsv_flag ~= 2
                    OITCZ_SMF_input.T_hat_obar_k_minus_deltabar = T_hat_obar{k-delta_bar+kIndexC}; % \hat{\mathcal{T}}_{k-\bar{\delta}}^{\bar{o}}, used in Line 9 of Algorithm 3
                    OITCZ_SMF_input.ell_k_minus_1 = ell(k-1+kIndexC); % Greatest maximum length up to k-1, used in Lines 11 and 13

                    if k > delta_bar
                        OITCZ_SMF_input.c_hat_obar_k = c_hat_obar(:, k+kIndexC); % Used in Lines 12 and 13: for k = d\bar{\delta}, it should be initialized as shown in Line 12.
                        OITCZ_SMF_input.d_inf_delta_bar = d_inf_delta_bar; % Used in Line 13

                        OITCZ_SMF_input.A_obar_power = A_obar_power; % For iteratively calculating the matrix power in Line 13
                    end
                end

                %-  Prior Refinement
                if OITCZ_SMF_input.prior_refinement_flag == 1    
                    if k >= 2 * delta_bar || OITCZ_SMF_input.initial_inclusion_flag == 1
                        OITCZ_SMF_input.range_for_refinement = IH_posterior_OITCZSMF{k-delta_bar+kIndexC}; % Providing the interval hull of the posterior can improve the efficiency (as we do not need to calculate another interval hull).
                    else % delta_bar <= k < 2 delta_bar, for invariance test
                        OITCZ_SMF_input.range_for_refinement = IH_posterior_OITCZSMF{k-delta_bar+kIndexC}; % Providing the interval hull of the posterior can improve the efficiency (as we do not need to calculate another interval hull).
                    end
                end

                OITCZ_SMF_output = OITCZ_SMF_EX(OITCZ_SMF_input);

                cZ_posterior_OITCZSMF{k+kIndexC} = OITCZ_SMF_output.cZ_posterior; % The estimate
                if obsv_flag ~= 2
                    T_hat_obar{k+kIndexC} = OITCZ_SMF_output.T_hat_obar_k; % \hat{\mathcal{T}}_k^{\bar{o}} in Line 13
                    ell(k+kIndexC) = OITCZ_SMF_output.ell_k; % Greatest maximum length up to k, in Line 11
                    c_hat_obar(:, k+1+kIndexC) = OITCZ_SMF_output.c_hat_obar_k_plus_1; % Used in Line 12

                    if k == delta_bar
                        d_inf_delta_bar = OITCZ_SMF_output.d_inf_delta_bar;
                    end

                    A_obar_power = OITCZ_SMF_output.A_obar_power; % For iteratively calculating the matrix power
                end
            end


            %%  Interval Hull of Estimate
            [zono_G_IH_OITCZSMF, zono_c_IH_OITCZSMF] = cZ_intervalhull(cZ_posterior_OITCZSMF{k+kIndexC});
            IH_posterior_OITCZSMF{k+kIndexC} = cZ_construct(eye(n), zono_c_IH_OITCZSMF, [], [], diag(zono_G_IH_OITCZSMF));
            
            zono_c_IH_OITCZSMF_real = zono_c_IH_OITCZSMF + x_sequence_difference(:, k+kIndexC); % Translate the state back
            IH_posterior_OITCZSMF_real{k+kIndexC} = cZ_construct(eye(n), zono_c_IH_OITCZSMF_real, [], [], diag(zono_G_IH_OITCZSMF));
            
            
            %%  Calculations
            %-  State inclusion
            if cZ_point_inclusion(cZ_posterior_OITCZSMF{k+kIndexC}, x_sequence(:, k+kIndexC)) == 0
                disp('The state is not in the estimate.')
                inclusion_indicators(k+kIndexC, n_trial, counter_n_o) = 0;
            end
            
            %-  Diameters - infinity norm
            diameters(k+kIndexC, n_trial, counter_n_o) = 2 * norm(diag(zono_G_IH_OITCZSMF));
            diameters_inf(k+kIndexC, n_trial, counter_n_o) = 2 * max(max(zono_G_IH_OITCZSMF));
            
            %-  Volumes
            volumes(k+kIndexC, n_trial, counter_n_o) = prod(2 * diag(zono_G_IH_OITCZSMF));
            
            %-  Bounds on estimation gaps
            componentwise_worst_case = zeros(n, 1);
            for i = 1: n
                componentwise_worst_case(i) = max(abs(x_sequence(i, k+kIndexC) - (zono_c_IH_OITCZSMF(i) + zono_G_IH_OITCZSMF(i, i))), abs(x_sequence(i, k+kIndexC) - (zono_c_IH_OITCZSMF(i) - zono_G_IH_OITCZSMF(i, i))));
            end
            bounds_estimation_gap(k+kIndexC, n_trial, counter_n_o) = norm(componentwise_worst_case);
            bounds_estimation_gap_inf(k+kIndexC, n_trial, counter_n_o) = max(componentwise_worst_case);
            
            %%  Highlighted Parts
            if n_trial <= num_highlights
                highlights{n_trial, counter_n_o}.x_sequence(:, k+kIndexC) = x_sequence(:, k+kIndexC);
                highlights{n_trial, counter_n_o}.x_sequence_difference(:, k+kIndexC) = x_sequence_difference(:, k+kIndexC);
                highlights{n_trial, counter_n_o}.y_sequence(:, k+kIndexC) = y_sequence(:, k+kIndexC);
                highlights{n_trial, counter_n_o}.cZ_posterior_OITCZSMF{k+kIndexC} = cZ_posterior_OITCZSMF{k+kIndexC};
                highlights{n_trial, counter_n_o}.IH_posterior_OITCZSMF{k+kIndexC} = IH_posterior_OITCZSMF{k+kIndexC};
                highlights{n_trial, counter_n_o}.IH_posterior_OITCZSMF_real{k+kIndexC} = IH_posterior_OITCZSMF_real{k+kIndexC};
            end
        end
    end
end
toc


%%  Calculations and Figures
diameters_average = zeros(1, k+kIndexC);
diameters_max = zeros(1, k+kIndexC);
diameters_min = inf(1, k+kIndexC);

diameters_inf_average = zeros(1, k+kIndexC);
diameters_inf_max = zeros(1, k+kIndexC);
diameters_inf_min = inf(1, k+kIndexC);

volumes_average = zeros(1, k+kIndexC);
volumes_max = zeros(1, k+kIndexC);
volumes_min = inf(1, k+kIndexC);

bounds_estimation_gap_average = zeros(1, k+kIndexC);
bounds_estimation_gap_max = zeros(1, k+kIndexC);
bounds_estimation_gap_min = inf(1, k+kIndexC);

bounds_estimation_gap_inf_average = zeros(1, k+kIndexC);
bounds_estimation_gap_inf_max = zeros(1, k+kIndexC);
bounds_estimation_gap_inf_min = inf(1, k+kIndexC);

for k = k_sequence
    for n_trial = 1: num_trials_each
        counter_n_o = 0;
        
        for n_o = set_of_n_o
            counter_n_o = counter_n_o + 1;
            
            diameters_average(k+kIndexC) = diameters_average(k+kIndexC) + diameters(k+kIndexC, n_trial, counter_n_o);
            diameters_max(k+kIndexC) = max(diameters_max(k+kIndexC), diameters(k+kIndexC, n_trial, counter_n_o));
            diameters_min(k+kIndexC) = min(diameters_min(k+kIndexC), diameters(k+kIndexC, n_trial, counter_n_o));
            
            diameters_inf_average(k+kIndexC) = diameters_inf_average(k+kIndexC) + diameters_inf(k+kIndexC, n_trial, counter_n_o);
            diameters_inf_max(k+kIndexC) = max(diameters_inf_max(k+kIndexC), diameters_inf(k+kIndexC, n_trial, counter_n_o));
            diameters_inf_min(k+kIndexC) = min(diameters_inf_min(k+kIndexC), diameters_inf(k+kIndexC, n_trial, counter_n_o));
            
            volumes_average(k+kIndexC) = volumes_average(k+kIndexC) + volumes(k+kIndexC, n_trial, counter_n_o);
            volumes_max(k+kIndexC) = max(volumes_max(k+kIndexC), volumes(k+kIndexC, n_trial, counter_n_o));
            volumes_min(k+kIndexC) = min(volumes_min(k+kIndexC), volumes(k+kIndexC, n_trial, counter_n_o));
            
            bounds_estimation_gap_average(k+kIndexC) = bounds_estimation_gap_average(k+kIndexC) + bounds_estimation_gap(k+kIndexC, n_trial, counter_n_o);
            bounds_estimation_gap_max(k+kIndexC) = max(bounds_estimation_gap_max(k+kIndexC), bounds_estimation_gap(k+kIndexC, n_trial, counter_n_o));
            bounds_estimation_gap_min(k+kIndexC) = min(bounds_estimation_gap_min(k+kIndexC), bounds_estimation_gap(k+kIndexC, n_trial, counter_n_o));
            
            bounds_estimation_gap_inf_average(k+kIndexC) = bounds_estimation_gap_inf_average(k+kIndexC) + bounds_estimation_gap_inf(k+kIndexC, n_trial, counter_n_o);
            bounds_estimation_gap_inf_max(k+kIndexC) = max(bounds_estimation_gap_inf_max(k+kIndexC), bounds_estimation_gap_inf(k+kIndexC, n_trial, counter_n_o));
            bounds_estimation_gap_inf_min(k+kIndexC) = min(bounds_estimation_gap_inf_min(k+kIndexC), bounds_estimation_gap_inf(k+kIndexC, n_trial, counter_n_o));
        end
    end
    
    diameters_average(k+kIndexC) = diameters_average(k+kIndexC) / (num_trials_each * counter_n_o);
    diameters_inf_average(k+kIndexC) = diameters_inf_average(k+kIndexC) / (num_trials_each * counter_n_o);
    volumes_average(k+kIndexC) = volumes_average(k+kIndexC) / (num_trials_each * counter_n_o);
    bounds_estimation_gap_average(k+kIndexC) = bounds_estimation_gap_average(k+kIndexC) / (num_trials_each * counter_n_o);
    bounds_estimation_gap_inf_average(k+kIndexC) = bounds_estimation_gap_inf_average(k+kIndexC) / (num_trials_each * counter_n_o);
end

figure,
plot(k_sequence, diameters_average, k_sequence, diameters_max, k_sequence, diameters_min)
xlabel('Time Step')
ylabel('Diameter')
grid on;

figure,
plot(k_sequence, diameters_inf_average, k_sequence, diameters_inf_max, k_sequence, diameters_inf_min)
xlabel('Time Step')
ylabel('Diameter - \infty-norm')
grid on;

figure,
plot(k_sequence, volumes_average, k_sequence, volumes_max, k_sequence, volumes_min)
xlabel('Time Step')
ylabel('Volume')
grid on;

figure,
plot(k_sequence, bounds_estimation_gap_average, k_sequence, bounds_estimation_gap_max, k_sequence, bounds_estimation_gap_min)
xlabel('Time Step')
ylabel('Bound on Estimation Gap')
grid on;

figure,
plot(k_sequence, bounds_estimation_gap_inf_average, k_sequence, bounds_estimation_gap_inf_max, k_sequence, bounds_estimation_gap_inf_min)
xlabel('Time Step')
ylabel('Bound on Estimation Gap - \infty-norm')
grid on;