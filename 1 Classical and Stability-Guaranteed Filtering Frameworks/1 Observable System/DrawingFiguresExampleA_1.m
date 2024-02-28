%%% Drawing Figures

close all;
clear;
clc;

load('experiment20240220')


%%  Figure 2
kSequence_selected = [0 1 2 3 4 5 6 10];

for k = kSequence_selected
    k
    
    if k == 0
        figure,
        plot(vertex_k_prior{k+kIndexC}(CH_k_prior{k+kIndexC}, 1), vertex_k_prior{k+kIndexC}(CH_k_prior{k+kIndexC}, 2), '-s', vertex_k_prior_A{k+kIndexC}(CH_k_prior_A{k+kIndexC}, 1), vertex_k_prior_A{k+kIndexC}(CH_k_prior_A{k+kIndexC}, 2), '-o',...
            vertex_k_prior_B{k+kIndexC}(CH_k_prior_B{k+kIndexC}, 1), vertex_k_prior_B{k+kIndexC}(CH_k_prior_B{k+kIndexC}, 2), '-*', vertex_k_prior_C{k+kIndexC}(CH_k_prior_C{k+kIndexC}, 1), vertex_k_prior_C{k+kIndexC}(CH_k_prior_C{k+kIndexC}, 2), '-^',...
            vertex_k_prior_D{k+kIndexC}(CH_k_prior_D{k+kIndexC}, 1), vertex_k_prior_D{k+kIndexC}(CH_k_prior_D{k+kIndexC}, 2), '-d', 'LineWidth', 1.2, 'MarkerSize', 8)
        legend('True', 'Alice', 'Bob', 'Carol', 'David')
        grid on;
        set(gca,'FontSize',12);
        xlabel('x^{(1)}-axis')
        ylabel('x^{(2)}-axis')
        axis([-4.5 4.5 -4.5 4.5])
    elseif k == 1
        figure,
        plot(vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 1), vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 2), '-s', vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 1), vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 2), '-o',...
                vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 1), vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 2), '-*', vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 1), vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 2), '-^',...
                vertex_k_posterior_D{k+kIndexC}(CH_k_posterior_D{k+kIndexC}, 1), vertex_k_posterior_D{k+kIndexC}(CH_k_posterior_D{k+kIndexC}, 2), '-d', 'LineWidth', 1.2, 'MarkerSize', 8)
        legend('True', 'Alice', 'Bob', 'Carol', 'David')
        grid on;
        set(gca,'FontSize',12);
        xlabel('x^{(1)}-axis')
        ylabel('x^{(2)}-axis')
%         axis([3.5 6.5 1 7])
    elseif k == 2
        figure,
        plot(vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 1), vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 2), '-s', vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 1), vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 2), '-o',...
            vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 1), vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 2), '-*', vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 1), vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 2), '-^',...
            NaN, NaN, '-d', 'LineWidth', 1.2, 'MarkerSize', 8)
        legend('True', 'Alice', 'Bob', 'Carol', 'David (empty)')
        grid on;
        set(gca,'FontSize',12);
        xlabel('x^{(1)}-axis')
        ylabel('x^{(2)}-axis')
%         axis([7.5 10.5 1.5 6.5])
        figure,
        plot(vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 1), vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 2), '-s', vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 1), vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 2), '-o',...
            vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 1), vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 2), '-*', vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 1), vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 2), '-^',...
            NaN, NaN, '-d', vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 1), vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 2), '--', 'LineWidth', 1.2, 'MarkerSize', 8)
        legend('True', 'Alice', 'Bob', 'Carol', 'David (empty)', 'OIT')
        grid on;
        set(gca,'FontSize',12);
        xlabel('x^{(1)}-axis')
        ylabel('x^{(2)}-axis')
    elseif k == 3
        figure,
        plot(vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 1), vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 2), '-s', vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 1), vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 2), '-o',...
            vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 1), vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 2), '-*', vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 1), vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 2), '-^',...
            NaN, NaN, '-d', vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 1), vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 2), '--', 'LineWidth', 1.2, 'MarkerSize', 8)
        legend('True', 'Alice', 'Bob', 'Carol', 'David (empty)', 'OIT')
        grid on;
        set(gca,'FontSize',12);
        xlabel('x^{(1)}-axis')
        ylabel('x^{(2)}-axis')
    elseif k == 4
        figure,
        plot(vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 1), vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 2), '-s', vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 1), vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 2), '-o',...
            vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 1), vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 2), '-*', vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 1), vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 2), '-^',...
            NaN, NaN, '-d', vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 1), vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 2), '--', 'LineWidth', 1.2, 'MarkerSize', 8)
        legend('True', 'Alice', 'Bob', 'Carol', 'David (empty)', 'OIT')
        grid on;
        set(gca,'FontSize',12);
        xlabel('x^{(1)}-axis')
        ylabel('x^{(2)}-axis')
    elseif k == 5
        figure,
        plot(vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 1), vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 2), '-s', vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 1), vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 2), '-o',...
            vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 1), vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 2), '-*', vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 1), vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 2), '-^',...
            NaN, NaN, '-d', vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 1), vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 2), '--', 'LineWidth', 1.2, 'MarkerSize', 8)
        legend('True', 'Alice', 'Bob', 'Carol', 'David (empty)', 'OIT')
        grid on;
        set(gca,'FontSize',12);
        xlabel('x^{(1)}-axis')
        ylabel('x^{(2)}-axis')
    elseif k == 6
        figure,
        plot(vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 1), vertex_k_posterior{k+kIndexC}(CH_k_posterior{k+kIndexC}, 2), '-s', vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 1), vertex_k_posterior_A{k+kIndexC}(CH_k_posterior_A{k+kIndexC}, 2), '-o',...
            vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 1), vertex_k_posterior_B{k+kIndexC}(CH_k_posterior_B{k+kIndexC}, 2), '-*', vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 1), vertex_k_posterior_C{k+kIndexC}(CH_k_posterior_C{k+kIndexC}, 2), '-^',...
            NaN, NaN, '-d', vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 1), vertex_k_OIT{k+kIndexC}(CH_k_OIT{k+kIndexC}, 2), '--', 'LineWidth', 1.2, 'MarkerSize', 8)
        legend('True', 'Alice', 'Bob', 'Carol', 'David (empty)', 'OIT')
        grid on;
        set(gca,'FontSize',12);
        xlabel('x^{(1)}-axis')
        ylabel('x^{(2)}-axis')
        xlim([25.5 28.5])
        ylim([2 7])
    end
end


%%  Maximum Diameters
MD = max(x_posterior_diameter(delta_optimal+kIndexC:end))
MD_A = max(x_posterior_diameter_A(delta_optimal+kIndexC:end))
MD_B = max(x_posterior_diameter_B(delta_optimal+kIndexC:end))
MD_C = max(x_posterior_diameter_C(delta_optimal+kIndexC:end))
diameter_upper_bound

k_draw_max = 50+kIndexC;

figure,
% plot(kSequence(1:k_draw_max), x_posterior_diameter(1:k_draw_max), '-s', kSequence(1:k_draw_max), x_posterior_diameter_A(1:k_draw_max), '-o', kSequence(1:k_draw_max), x_posterior_diameter_B(1:k_draw_max), '-*', kSequence(1:k_draw_max), x_posterior_diameter_C(1:k_draw_max), '-^', [min(kSequence(2:k_draw_max)) max(kSequence(2:k_draw_max))], [diameter_upper_bound, diameter_upper_bound], '-', 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), x_posterior_diameter(1:k_draw_max), '-s', kSequence(1:k_draw_max), x_posterior_diameter_A(1:k_draw_max), '-o', kSequence(1:k_draw_max), x_posterior_diameter_B(1:k_draw_max), '-*',...
    kSequence(1:k_draw_max), x_posterior_diameter_C(1:k_draw_max), '-^', 'LineWidth', 1.2, 'MarkerSize', 6)
hold on;
plot(kSequence(delta:k_draw_max), x_OIT_diameter(delta:k_draw_max), '-', 'LineWidth', 2)
plot([delta delta], [0 9], '-.')
legend('True', 'Alice', 'Bob', 'Carol', 'OIT')
grid on;
set(gca,'FontSize',12);
ylim([0 9])
xlabel('Time Step')
ylabel('Diameter')

figure,
plot(kSequence(1:k_draw_max), x_posterior_estimation_gap_A(1:k_draw_max), '-o', 'Color', [0.85,0.33,0.10], 'LineWidth', 1.2, 'MarkerSize', 6)
hold on;
plot(kSequence(1:k_draw_max), x_posterior_estimation_gap_B(1:k_draw_max), '-*', 'Color', [0.93,0.69,0.13], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), x_posterior_estimation_gap_C(1:k_draw_max), '-^', 'Color', [0.49,0.18,0.56], 'LineWidth', 1.2, 'MarkerSize', 6)
legend('Alice', 'Bob', 'Carol')
title('Estimation Gap')
grid on;
set(gca,'FontSize',12);
xlabel('Time Step')
ylabel('Estimation Gap')


%%  Component-Wise Intervals
posterior_k_max = zeros(2, kmax+kIndexC);
posterior_k_min = zeros(2, kmax+kIndexC);
posterior_k_max_A = zeros(2, kmax+kIndexC);
posterior_k_min_A = zeros(2, kmax+kIndexC);
posterior_k_max_B = zeros(2, kmax+kIndexC);
posterior_k_min_B = zeros(2, kmax+kIndexC);
posterior_k_max_C = zeros(2, kmax+kIndexC);
posterior_k_min_C = zeros(2, kmax+kIndexC);

for k = kSequence
    for i = 1: 2
        posterior_k_max(i, k+kIndexC) = max(vertex_k_posterior{k+kIndexC}(:, i));
        posterior_k_min(i, k+kIndexC) = min(vertex_k_posterior{k+kIndexC}(:, i));
        
        posterior_k_max_A(i, k+kIndexC) = max(vertex_k_posterior_A{k+kIndexC}(:, i));
        posterior_k_min_A(i, k+kIndexC) = min(vertex_k_posterior_A{k+kIndexC}(:, i));
        
        posterior_k_max_B(i, k+kIndexC) = max(vertex_k_posterior_B{k+kIndexC}(:, i));
        posterior_k_min_B(i, k+kIndexC) = min(vertex_k_posterior_B{k+kIndexC}(:, i));
        
        posterior_k_max_C(i, k+kIndexC) = max(vertex_k_posterior_C{k+kIndexC}(:, i));
        posterior_k_min_C(i, k+kIndexC) = min(vertex_k_posterior_C{k+kIndexC}(:, i));
    end
end

figure,
plot(kSequence(1:k_draw_max), xSequence(1, 1:k_draw_max), '-', 'Color', [0.30,0.75,0.93], 'LineWidth', 1.2, 'MarkerSize', 6)
hold on;
plot(kSequence(1:k_draw_max), posterior_k_max(1, 1:k_draw_max), '-s', 'Color', [0.00,0.45,0.74], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_max_A(1, 1:k_draw_max), '-o', 'Color', [0.85,0.33,0.10], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_max_B(1, 1:k_draw_max), '-*', 'Color', [0.93,0.69,0.13], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_max_C(1, 1:k_draw_max), '-^', 'Color', [0.49,0.18,0.56], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_min(1, 1:k_draw_max), '-s', 'Color', [0.00,0.45,0.74], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_min_A(1, 1:k_draw_max), '-o', 'Color', [0.85,0.33,0.10], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_min_B(1, 1:k_draw_max), '-*', 'Color', [0.93,0.69,0.13], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_min_C(1, 1:k_draw_max), '-^', 'Color', [0.49,0.18,0.56], 'LineWidth', 1.2, 'MarkerSize', 6)
xlabel('Time Step')
ylabel('x^{(1)}')
legend('State trajectory', 'True', 'Alice', 'Bob', 'Carol')
grid on;
set(gca,'FontSize',12);
ylim([-1 310])

axes('position', [0.172619047619047,0.499206349206349,0.377380952380953,0.412698412698429])
box on % put box around new pair of axes
k_draw_max_box = 5;
plot(kSequence(1:k_draw_max_box), xSequence(1, 1:k_draw_max_box), '-', 'Color', [0.30,0.75,0.93], 'LineWidth', 1.2, 'MarkerSize', 6)
hold on;
plot(kSequence(1:k_draw_max_box), posterior_k_max(1, 1:k_draw_max_box), '-s', 'Color', [0.00,0.45,0.74], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max_box), posterior_k_max_A(1, 1:k_draw_max_box), '-o', 'Color', [0.85,0.33,0.10], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max_box), posterior_k_max_B(1, 1:k_draw_max_box), '-*', 'Color', [0.93,0.69,0.13], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max_box), posterior_k_max_C(1, 1:k_draw_max_box), '-^', 'Color', [0.49,0.18,0.56], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max_box), posterior_k_min(1, 1:k_draw_max_box), '-s', 'Color', [0.00,0.45,0.74], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max_box), posterior_k_min_A(1, 1:k_draw_max_box), '-o', 'Color', [0.85,0.33,0.10], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max_box), posterior_k_min_B(1, 1:k_draw_max_box), '-*', 'Color', [0.93,0.69,0.13], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max_box), posterior_k_min_C(1, 1:k_draw_max_box), '-^', 'Color', [0.49,0.18,0.56], 'LineWidth', 1.2, 'MarkerSize', 6)
axis tight
grid on;
set(gca,'FontSize',12);


figure,
plot(kSequence(1:k_draw_max), xSequence(2, 1:k_draw_max), '-', 'Color', [0.30,0.75,0.93], 'LineWidth', 1.2, 'MarkerSize', 6)
hold on;
plot(kSequence(1:k_draw_max), posterior_k_max(2, 1:k_draw_max), '-s', 'Color', [0.00,0.45,0.74], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_max_A(2, 1:k_draw_max), '-o', 'Color', [0.85,0.33,0.10], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_max_B(2, 1:k_draw_max), '-*', 'Color', [0.93,0.69,0.13], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_max_C(2, 1:k_draw_max), '-^', 'Color', [0.49,0.18,0.56], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_min_A(2, 1:k_draw_max), '-o', 'Color', [0.85,0.33,0.10], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_min_A(2, 1:k_draw_max), '-o', 'Color', [0.85,0.33,0.10], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_min_B(2, 1:k_draw_max), '-*', 'Color', [0.93,0.69,0.13], 'LineWidth', 1.2, 'MarkerSize', 6)
plot(kSequence(1:k_draw_max), posterior_k_min_C(2, 1:k_draw_max), '-^', 'Color', [0.49,0.18,0.56], 'LineWidth', 1.2, 'MarkerSize', 6)
xlabel('Time Step')
ylabel('x^{(2)}')
legend('State trajectory', 'True', 'Alice', 'Bob', 'Carol')
grid on;
set(gca,'FontSize',12);