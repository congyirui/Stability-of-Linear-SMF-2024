%%% Example B - 3 - Compared with CZ-SMFs
%   Initial condition = the true initial range
%   Toolbox CORA 2024 is needed by CZ-SMF.
%
%   (c) Yirui Cong, created: 22-Mar-2021, last modified: 28-Feb-2024

close all;
clear;
clc;


%%  Simulation Parameters
k_max = 100;
k_sequence = 0: k_max;
kIndexC = 1; % A compensator for the index 0 in matlab: for example, y_0 in matlab is y(:,1) = y(0 + k_indexC).

set_of_n = [10 10 20 30]; % There are some additional costs (at 0.1ms level) on the first n (the reason is not clear), and thus cannot be neglected in Algorithm 3 (since it is at 0.1ms level).
                          % So, n = 10 is duplicated.
length_set_of_n = length(set_of_n);

num_trials_each = 100; % Number of trials for each beta

CZ_SMF_nc = 1; % Maximum number of constraints for the reduction
CZ_SMF_od = 1; % Degree-of-freedom order


%%  Trials
running_time_description_OITCZSMF = zeros(1, length_set_of_n);
running_time_description_CZSMF_g = zeros(1, length_set_of_n); % With the reduction method 'girard'
running_time_description_CZSMF_c = zeros(1, length_set_of_n); % With the reduction method 'combastel'

diameters_OITCZSMF = zeros(k_max+kIndexC, num_trials_each, length_set_of_n);
diameters_CZSMF_g = zeros(k_max+kIndexC, num_trials_each, length_set_of_n);
diameters_CZSMF_c = zeros(k_max+kIndexC, num_trials_each, length_set_of_n);

bounds_estimation_gap_OITCZSMF = zeros(k_max+kIndexC, num_trials_each, length_set_of_n);
bounds_estimation_gap_CZSMF_g = zeros(k_max+kIndexC, num_trials_each, length_set_of_n);
bounds_estimation_gap_CZSMF_c = zeros(k_max+kIndexC, num_trials_each, length_set_of_n);

counter_n = 0;
for n = set_of_n
    counter_n = counter_n + 1;
    
    n_o = n; % Observable systems
    n_obar = n - n_o;

    p = n_o;
    m = n_o;
    
    for n_trial = 1: num_trials_each
        n_trial

        %%  System Parameters
        flag_successfully_generated = 0;

        while flag_successfully_generated == 0
            %-  Observable Subsystem
            sys = drss(n_o, m, p);

            A_o = sys.A;
            B_o = sys.B;
            C_o = sys.C;

            obsv_flag = 2;

            if rank(obsv(A_o, C_o)) == n
                flag_successfully_generated = 1;

                A = A_o;
                B = B_o;
                C = C_o;
            else
                continue; % The numerical precision might not be high enough to support a bounded estimate.
            end
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

        %-  Parameters of Actual Range of the Initial State
        G_cZ_0_real = 10 * eye(n);
        c_cZ_0_real = zeros(n, 1);

        % cZ_G_0_real = 10 * eye(n);
        % cZ_c_0_real = zeros(n, 1) + 100;


       %%  Initialization of OIT-CZ SMF (Line 1 in Algorithm 3)
        G_cZ_0_prior_OITCZSMF = G_cZ_0_real;
        c_cZ_0_prior_OITCZSMF = c_cZ_0_real;
    %         c_cZ_0_prior_OITCZSMF = cZ_c_0_real + 10 * 2 * (rand(n, 1) - 0.5);
        % cZ_c_0_prior_OITCZSMF = cZ_c_0_real + 10;
        A_cZ_0_prior_OITCZSMF = [];
        b_cZ_0_prior_OITCZSMF = [];
        cwb_cZ_0_prior_OITCZSMF = ones(size(G_cZ_0_real, 2), 1);
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

        %- Unobservale part of initial range (used in Line 5)
        if obsv_flag ~= 2
            G_cZ_temp = P * G_cZ_0_prior_OITCZSMF;
            c_cZ_temp = P * c_cZ_0_prior_OITCZSMF;
            cZ_prior_uos_OITCZSMF_0 = cZ_construct(G_cZ_temp(n_o+1: end, :), c_cZ_temp(n_o+1: end), A_cZ_0_prior_OITCZSMF, b_cZ_0_prior_OITCZSMF, cwb_cZ_0_prior_OITCZSMF);
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
        OITCZ_SMF_input.prior_refinement_flag = 0;
        OITCZ_SMF_input.initial_inclusion_flag = 1;

        OITCZ_SMF_input.column_scaling_flag = 0;

        if obsv_flag ~= 2
            OITCZ_SMF_input.A_o = A_o;
            OITCZ_SMF_input.P = P;
            OITCZ_SMF_input.A_obar = A_obar;
            OITCZ_SMF_input.A_21 = A_21;
            OITCZ_SMF_input.B_obar = B_obar;

            OITCZ_SMF_input.B_o = B_o; % For test
            OITCZ_SMF_input.C_o = C_o; % For test

            OITCZ_SMF_input.epsilon = epsilon;
            OITCZ_SMF_input.Upsilon_inf = Upsilon_inf;
        end


        %%  Initialization of CZ-SMF with 'girard'
        G_cZ_0_prior_CZSMF_g = G_cZ_0_real;
        c_cZ_0_prior_CZSMF_g = c_cZ_0_real;
        A_cZ_0_prior_CZSMF_g = [];
        b_cZ_0_prior_CZSMF_g = [];
        cwb_cZ_0_prior_CZSMF_g = ones(size(G_cZ_0_real, 2), 1);
        cZ_prior_CZSMF_g_0 = cZ_construct(G_cZ_0_prior_CZSMF_g, c_cZ_0_prior_CZSMF_g, A_cZ_0_prior_CZSMF_g, b_cZ_0_prior_CZSMF_g, cwb_cZ_0_prior_CZSMF_g);
        
        
        %%  Initialization of CZ-SMF with 'combastel'
        G_cZ_0_prior_CZSMF_c = G_cZ_0_real;
        c_cZ_0_prior_CZSMF_c = c_cZ_0_real;
        A_cZ_0_prior_CZSMF_c = [];
        b_cZ_0_prior_CZSMF_c = [];
        cwb_cZ_0_prior_CZSMF_c = ones(size(G_cZ_0_real, 2), 1);
        cZ_prior_CZSMF_c_0 = cZ_construct(G_cZ_0_prior_CZSMF_c, c_cZ_0_prior_CZSMF_c, A_cZ_0_prior_CZSMF_c, b_cZ_0_prior_CZSMF_c, cwb_cZ_0_prior_CZSMF_c);


        %%  Data Storage
        % cZ_prior_OITCZSMF = cell(k_max+kIndexC, 1); % Prior
        % cZ_prior_OITCZSMF{k+kIndexC} = cZ_prior_OITCZSMF_0;
        cZ_posterior_OITCZSMF = cell(k_max+kIndexC, 1); % Posterior

        if obsv_flag ~= 2
            T_hat_obar = cell(k_max+kIndexC, 1); % Store \hat{\mathcal{T}}_k^{\bar{o}}
            ell = zeros(1, k_max+kIndexC);
            c_hat_obar = zeros(n_obar, k_max+kIndexC+1);
        end

        cZ_posterior_CZSMF_g = cell(k_max+kIndexC, 1); % Posterior for CZ-SMF with 'girard'
        cZ_posterior_CZSMF_c = cell(k_max+kIndexC, 1); % Posterior for CZ-SMF with 'combastel'

        x_sequence = zeros(n, k_max+kIndexC);
        y_sequence = zeros(m, k_max+kIndexC);
        w_sequence = cZ_w.G * 2 * (rand(p, k_max+kIndexC) - 0.5) + cZ_w.c; % Process noises
        v_sequence = cZ_v.G * 2 * (rand(m, k_max+kIndexC) - 0.5) + cZ_v.c; % Measurement noises


        %%  Simulations
        for k = k_sequence
            k
            OITCZ_SMF_input.k = k;


            %%  Realizations of States and Measurements
            if k == 0
                x_sequence(:, kIndexC) = G_cZ_0_real * 2 * (rand(n, 1) - 0.5) + c_cZ_0_real;
            else
                x_sequence(:, k+kIndexC) = A * x_sequence(:, k-1+kIndexC) + B * w_sequence(:, k-1+kIndexC);
            end

            y_sequence(:, k+kIndexC) = C * x_sequence(:, k+kIndexC) + v_sequence(:, k+kIndexC);


            %%  OIT-CZ SMF
            tic
            if k < delta_bar
                OITCZ_SMF_input.y_sequence = y_sequence(:, kIndexC: k+kIndexC);
                if k == 0
                    OITCZ_SMF_input.cZ_prior_0 = cZ_prior_OITCZSMF_0; % Initial prior range
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
            else
                OITCZ_SMF_input.y_sequence = y_sequence(:, k-delta_bar+kIndexC: k+kIndexC); % y_{k-delta}, ..., y_k 
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
                        OITCZ_SMF_input.range_for_refinement = IH_posterior_OITCZSMF{k-delta_bar+kIndexC};
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
            toc
            running_time_description_OITCZSMF(counter_n) = running_time_description_OITCZSMF(counter_n) + toc;


            %%  CZ-SMF with 'girard'
            tic
            if k == 0
                cZ_in_CZSMF_g = cZ_prior_CZSMF_g_0;
            else
                cZ_in_CZSMF_g = cZ_posterior_CZSMF_g{k-1+kIndexC};
            end

            cZ_posterior_CZSMF_g{k+kIndexC} = CZ_SMF(A, B, C, cZ_in_CZSMF_g, y_sequence(:, k+kIndexC), k, cZ_w, cZ_v, 'girard', CZ_SMF_nc, CZ_SMF_od);
            toc
            running_time_description_CZSMF_g(counter_n) = running_time_description_CZSMF_g(counter_n) + toc;
            
            
            %%  CZ-SMF with 'combastel'
            tic
            if k == 0
                cZ_in_CZSMF_c = cZ_prior_CZSMF_c_0;
            else
                cZ_in_CZSMF_c = cZ_posterior_CZSMF_c{k-1+kIndexC};
            end

            cZ_posterior_CZSMF_c{k+kIndexC} = CZ_SMF(A, B, C, cZ_in_CZSMF_c, y_sequence(:, k+kIndexC), k, cZ_w, cZ_v, 'combastel', CZ_SMF_nc, CZ_SMF_od);
            toc
            running_time_description_CZSMF_c(counter_n) = running_time_description_CZSMF_c(counter_n) + toc;
            
            
            %%  Interval Hull of Estimate
            [zono_G_IH_OITCZSMF, zono_c_IH_OITCZSMF] = cZ_intervalhull(cZ_posterior_OITCZSMF{k+kIndexC});
            IH_posterior_OITCZSMF{k+kIndexC} = cZ_construct(eye(n), zono_c_IH_OITCZSMF, [], [], diag(zono_G_IH_OITCZSMF));
            
            [zono_G_IH_CZSMF_g, zono_c_IH_CZSMF_g] = cZ_intervalhull(cZ_posterior_CZSMF_g{k+kIndexC});
            IH_posterior_CZSMF_g{k+kIndexC} = cZ_construct(zono_G_IH_CZSMF_g, zono_c_IH_CZSMF_g, [], [], ones(n, 1));
            
            [zono_G_IH_CZSMF_c, zono_c_IH_CZSMF_c] = cZ_intervalhull(cZ_posterior_CZSMF_c{k+kIndexC});
            IH_posterior_CZSMF_c{k+kIndexC} = cZ_construct(zono_G_IH_CZSMF_c, zono_c_IH_CZSMF_c, [], [], ones(n, 1));
            
            
            %%  Calculations
            %-  Diameters - infinity norm
            diameters_OITCZSMF(k+kIndexC, n_trial, counter_n) = max(max(zono_G_IH_OITCZSMF));
            
            diameters_CZSMF_g(k+kIndexC, n_trial, counter_n) = max(max(zono_G_IH_CZSMF_g));
            
            diameters_CZSMF_c(k+kIndexC, n_trial, counter_n) = max(max(zono_G_IH_CZSMF_c));
            
            %-  Bounds on estimation gaps
            componentwise_worst_case = zeros(n, 1);
            for i = 1: n
                componentwise_worst_case(i) = max(abs(x_sequence(i, k+kIndexC) - (zono_c_IH_OITCZSMF(i) + zono_G_IH_OITCZSMF(i, i))), abs(x_sequence(i, k+kIndexC) - (zono_c_IH_OITCZSMF(i) - zono_G_IH_OITCZSMF(i, i))));
            end
            bounds_estimation_gap_OITCZSMF(k+kIndexC, n_trial, counter_n) = norm(componentwise_worst_case);
            
            componentwise_worst_case = zeros(n, 1);
            for i = 1: n
                componentwise_worst_case(i) = max(abs(x_sequence(i, k+kIndexC) - (zono_c_IH_CZSMF_g(i) + zono_G_IH_CZSMF_g(i, i))), abs(x_sequence(i, k+kIndexC) - (zono_c_IH_CZSMF_g(i) - zono_G_IH_CZSMF_g(i, i))));
            end
            bounds_estimation_gap_CZSMF_g(k+kIndexC, n_trial, counter_n) = norm(componentwise_worst_case);
            
            componentwise_worst_case = zeros(n, 1);
            for i = 1: n
                componentwise_worst_case(i) = max(abs(x_sequence(i, k+kIndexC) - (zono_c_IH_CZSMF_c(i) + zono_G_IH_CZSMF_c(i, i))), abs(x_sequence(i, k+kIndexC) - (zono_c_IH_CZSMF_c(i) - zono_G_IH_CZSMF_c(i, i))));
            end
            bounds_estimation_gap_CZSMF_c(k+kIndexC, n_trial, counter_n) = norm(componentwise_worst_case);
        end
    end
end


%%  Calculations and Figures
averaged_running_time_description_OITCZSMF = running_time_description_OITCZSMF / ((k_max + 1) * num_trials_each)
averaged_running_time_description_CZSMF_g = running_time_description_CZSMF_g / ((k_max + 1) * num_trials_each)
averaged_running_time_description_CZSMF_c = running_time_description_CZSMF_c / ((k_max + 1) * num_trials_each)

diameters_average_OITCZSMF = zeros(length_set_of_n, k+kIndexC);
diameters_max_OITCZSMF = zeros(length_set_of_n, k+kIndexC);
diameters_min_OITCZSMF = inf(length_set_of_n, k+kIndexC);

diameters_average_CZSMF_g = zeros(length_set_of_n, k+kIndexC);
diameters_max_CZSMF_g = zeros(length_set_of_n, k+kIndexC);
diameters_min_CZSMF_g = inf(length_set_of_n, k+kIndexC);

diameters_average_CZSMF_c = zeros(length_set_of_n, k+kIndexC);
diameters_max_CZSMF_c = zeros(length_set_of_n, k+kIndexC);
diameters_min_CZSMF_c = inf(length_set_of_n, k+kIndexC);

bounds_estimation_gap_average_OITCZSMF = zeros(length_set_of_n, k+kIndexC);
bounds_estimation_gap_max_OITCZSMF = zeros(length_set_of_n, k+kIndexC);
bounds_estimation_gap_min_OITCZSMF = inf(length_set_of_n, k+kIndexC);

bounds_estimation_gap_average_CZSMF_g = zeros(length_set_of_n, k+kIndexC);
bounds_estimation_gap_max_CZSMF_g = zeros(length_set_of_n, k+kIndexC);
bounds_estimation_gap_min_CZSMF_g = inf(length_set_of_n, k+kIndexC);

bounds_estimation_gap_average_CZSMF_c = zeros(length_set_of_n, k+kIndexC);
bounds_estimation_gap_max_CZSMF_c = zeros(length_set_of_n, k+kIndexC);
bounds_estimation_gap_min_CZSMF_c = inf(length_set_of_n, k+kIndexC);

for k = k_sequence
    for n_trial = 1: num_trials_each
        counter_n = 0;
        
        for n = set_of_n
            counter_n = counter_n + 1;
            
            diameters_average_OITCZSMF(counter_n, k+kIndexC) = diameters_average_OITCZSMF(counter_n, k+kIndexC) + diameters_OITCZSMF(k+kIndexC, n_trial, counter_n);
            diameters_max_OITCZSMF(counter_n, k+kIndexC) = max(diameters_max_OITCZSMF(counter_n, k+kIndexC), diameters_OITCZSMF(k+kIndexC, n_trial, counter_n));
            diameters_min_OITCZSMF(counter_n, k+kIndexC) = min(diameters_min_OITCZSMF(counter_n, k+kIndexC), diameters_OITCZSMF(k+kIndexC, n_trial, counter_n));
            
            diameters_average_CZSMF_g(counter_n, k+kIndexC) = diameters_average_CZSMF_g(counter_n, k+kIndexC) + diameters_CZSMF_g(k+kIndexC, n_trial, counter_n);
            diameters_max_CZSMF_g(counter_n, k+kIndexC) = max(diameters_max_CZSMF_g(counter_n, k+kIndexC), diameters_CZSMF_g(k+kIndexC, n_trial, counter_n));
            diameters_min_CZSMF_g(counter_n, k+kIndexC) = min(diameters_min_CZSMF_g(counter_n, k+kIndexC), diameters_CZSMF_g(k+kIndexC, n_trial, counter_n));
            
            diameters_average_CZSMF_c(counter_n, k+kIndexC) = diameters_average_CZSMF_c(counter_n, k+kIndexC) + diameters_CZSMF_c(k+kIndexC, n_trial, counter_n);
            diameters_max_CZSMF_c(counter_n, k+kIndexC) = max(diameters_max_CZSMF_c(counter_n, k+kIndexC), diameters_CZSMF_c(k+kIndexC, n_trial, counter_n));
            diameters_min_CZSMF_c(counter_n, k+kIndexC) = min(diameters_min_CZSMF_c(counter_n, k+kIndexC), diameters_CZSMF_c(k+kIndexC, n_trial, counter_n));
            
            bounds_estimation_gap_average_OITCZSMF(counter_n, k+kIndexC) = bounds_estimation_gap_average_OITCZSMF(counter_n, k+kIndexC) + bounds_estimation_gap_OITCZSMF(k+kIndexC, n_trial, counter_n);
            bounds_estimation_gap_max_OITCZSMF(counter_n, k+kIndexC) = max(bounds_estimation_gap_max_OITCZSMF(counter_n, k+kIndexC), bounds_estimation_gap_OITCZSMF(k+kIndexC, n_trial, counter_n));
            bounds_estimation_gap_min_OITCZSMF(counter_n, k+kIndexC) = min(bounds_estimation_gap_min_OITCZSMF(counter_n, k+kIndexC), bounds_estimation_gap_OITCZSMF(k+kIndexC, n_trial, counter_n));
            
            bounds_estimation_gap_average_CZSMF_g(counter_n, k+kIndexC) = bounds_estimation_gap_average_CZSMF_g(counter_n, k+kIndexC) + bounds_estimation_gap_CZSMF_g(k+kIndexC, n_trial, counter_n);
            bounds_estimation_gap_max_CZSMF_g(counter_n, k+kIndexC) = max(bounds_estimation_gap_max_CZSMF_g(counter_n, k+kIndexC), bounds_estimation_gap_CZSMF_g(k+kIndexC, n_trial, counter_n));
            bounds_estimation_gap_min_CZSMF_g(counter_n, k+kIndexC) = min(bounds_estimation_gap_min_CZSMF_g(counter_n, k+kIndexC), bounds_estimation_gap_CZSMF_g(k+kIndexC, n_trial, counter_n));
            
            bounds_estimation_gap_average_CZSMF_c(counter_n, k+kIndexC) = bounds_estimation_gap_average_CZSMF_c(counter_n, k+kIndexC) + bounds_estimation_gap_CZSMF_c(k+kIndexC, n_trial, counter_n);
            bounds_estimation_gap_max_CZSMF_c(counter_n, k+kIndexC) = max(bounds_estimation_gap_max_CZSMF_c(counter_n, k+kIndexC), bounds_estimation_gap_CZSMF_c(k+kIndexC, n_trial, counter_n));
            bounds_estimation_gap_min_CZSMF_c(counter_n, k+kIndexC) = min(bounds_estimation_gap_min_CZSMF_c(counter_n, k+kIndexC), bounds_estimation_gap_CZSMF_c(k+kIndexC, n_trial, counter_n));
        end
    end
end

diameters_average_OITCZSMF = diameters_average_OITCZSMF / num_trials_each;
bounds_estimation_gap_average_OITCZSMF = bounds_estimation_gap_average_OITCZSMF / num_trials_each;

diameters_average_CZSMF_g = diameters_average_CZSMF_g / num_trials_each;
bounds_estimation_gap_average_CZSMF_g = bounds_estimation_gap_average_CZSMF_g / num_trials_each;

diameters_average_CZSMF_c = diameters_average_CZSMF_c / num_trials_each;
bounds_estimation_gap_average_CZSMF_c = bounds_estimation_gap_average_CZSMF_c / num_trials_each;

diameters_average_OITCZSMF_total = sum(diameters_average_OITCZSMF, 2) / (k_max + kIndexC)
bounds_estimation_gap_average_OITCZSMF_total = sum(bounds_estimation_gap_average_OITCZSMF, 2) / (k_max + kIndexC);

diameters_average_CZSMF_g_total = sum(diameters_average_CZSMF_g, 2) / (k_max + kIndexC)
bounds_estimation_gap_average_CZSMF_g_total = sum(bounds_estimation_gap_average_CZSMF_g, 2) / (k_max + kIndexC);

diameters_average_CZSMF_c_total = sum(diameters_average_CZSMF_c, 2) / (k_max + kIndexC)
bounds_estimation_gap_average_CZSMF_c_total = sum(bounds_estimation_gap_average_CZSMF_c, 2) / (k_max + kIndexC);

figure,
plot(k_sequence, diameters_average_OITCZSMF(2, :), k_sequence, diameters_max_OITCZSMF(2, :), k_sequence, diameters_min_OITCZSMF(2, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('OIT-CZ SMF (n = 10)')

figure,
plot(k_sequence, diameters_average_CZSMF_g(2, :), k_sequence, diameters_max_CZSMF_g(2, :), k_sequence, diameters_min_CZSMF_g(2, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('CZ-SMF (girard, n = 10)')

figure,
plot(k_sequence, diameters_average_CZSMF_c(2, :), k_sequence, diameters_max_CZSMF_c(2, :), k_sequence, diameters_min_CZSMF_c(2, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('CZ-SMF (combastel, n = 10)')

% figure,
% plot(k_sequence, bounds_estimation_gap_average_OITCZSMF(2, :), k_sequence, bounds_estimation_gap_max_OITCZSMF(2, :), k_sequence, bounds_estimation_gap_min_OITCZSMF(2, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('OIT-CZ SMF (n = 10)')
% 
% figure,
% plot(k_sequence, bounds_estimation_gap_average_CZSMF_g(2, :), k_sequence, bounds_estimation_gap_max_CZSMF_g(2, :), k_sequence, bounds_estimation_gap_min_CZSMF_g(2, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('CZ-SMF (girard, n = 10)')
% 
% figure,
% plot(k_sequence, bounds_estimation_gap_average_CZSMF_c(2, :), k_sequence, bounds_estimation_gap_max_CZSMF_c(2, :), k_sequence, bounds_estimation_gap_min_CZSMF_c(2, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('CZ-SMF (combastel, n = 10)')

figure,
plot(k_sequence, diameters_average_OITCZSMF(3, :), k_sequence, diameters_max_OITCZSMF(3, :), k_sequence, diameters_min_OITCZSMF(3, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('OIT-CZ SMF (n = 20)')

figure,
plot(k_sequence, diameters_average_CZSMF_g(3, :), k_sequence, diameters_max_CZSMF_g(3, :), k_sequence, diameters_min_CZSMF_g(3, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('CZ-SMF (girard, n = 20)')

figure,
plot(k_sequence, diameters_average_CZSMF_c(3, :), k_sequence, diameters_max_CZSMF_c(3, :), k_sequence, diameters_min_CZSMF_c(3, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('CZ-SMF (combastel, n = 20)')

% figure,
% plot(k_sequence, bounds_estimation_gap_average_OITCZSMF(3, :), k_sequence, bounds_estimation_gap_max_OITCZSMF(3, :), k_sequence, bounds_estimation_gap_min_OITCZSMF(3, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('OIT-CZ SMF (n = 20)')
% 
% figure,
% plot(k_sequence, bounds_estimation_gap_average_CZSMF_g(3, :), k_sequence, bounds_estimation_gap_max_CZSMF_g(3, :), k_sequence, bounds_estimation_gap_min_CZSMF_g(3, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('CZ-SMF (girard, n = 20)')
% 
% figure,
% plot(k_sequence, bounds_estimation_gap_average_CZSMF_c(3, :), k_sequence, bounds_estimation_gap_max_CZSMF_c(3, :), k_sequence, bounds_estimation_gap_min_CZSMF_c(3, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('CZ-SMF (combastel, n = 20)')

figure,
plot(k_sequence, diameters_average_OITCZSMF(4, :), k_sequence, diameters_max_OITCZSMF(4, :), k_sequence, diameters_min_OITCZSMF(4, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('OIT-CZ SMF (n = 30)')

figure,
plot(k_sequence, diameters_average_CZSMF_g(4, :), k_sequence, diameters_max_CZSMF_g(4, :), k_sequence, diameters_min_CZSMF_g(4, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('CZ-SMF (girard, n = 30)')

figure,
plot(k_sequence, diameters_average_CZSMF_c(4, :), k_sequence, diameters_max_CZSMF_c(4, :), k_sequence, diameters_min_CZSMF_c(4, :))
xlabel('Time Step')
ylabel('Diameter')
legend('average', 'max', 'min')
grid on;
title('CZ-SMF (combastel, n = 30)')

% figure,
% plot(k_sequence, bounds_estimation_gap_average_OITCZSMF(4, :), k_sequence, bounds_estimation_gap_max_OITCZSMF(4, :), k_sequence, bounds_estimation_gap_min_OITCZSMF(4, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('OIT-CZ SMF (n = 30)')
% 
% figure,
% plot(k_sequence, bounds_estimation_gap_average_CZSMF_g(4, :), k_sequence, bounds_estimation_gap_max_CZSMF_g(4, :), k_sequence, bounds_estimation_gap_min_CZSMF_g(4, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('CZ-SMF (girard, n = 30)')
% 
% figure,
% plot(k_sequence, bounds_estimation_gap_average_CZSMF_c(4, :), k_sequence, bounds_estimation_gap_max_CZSMF_c(4, :), k_sequence, bounds_estimation_gap_min_CZSMF_c(4, :))
% xlabel('Time Step')
% ylabel('Bound on Estimation Gap')
% legend('average', 'max', 'min')
% grid on;
% title('CZ-SMF (combastel, n = 30)')