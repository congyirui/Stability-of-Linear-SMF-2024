%%% Drawing Figures

close all;
clear;
clc;

load('experiment_ds_20240218') % The data is too large. One can run "ExampleB_1_DS" to derive the same data.


%%  Figure 4(b)
k_sequence_drawing = k_sequence;

diameters_average_drawing = diameters_inf_average(k_sequence_drawing + kIndexC);
diameters_max_drawing = diameters_inf_max(k_sequence_drawing + kIndexC);
diameters_min_drawing = diameters_inf_min(k_sequence_drawing + kIndexC);

bounds_estimation_gap_average_drawing = bounds_estimation_gap_inf_average(k_sequence_drawing + kIndexC);
bounds_estimation_gap_max_drawing = bounds_estimation_gap_inf_max(k_sequence_drawing + kIndexC);
bounds_estimation_gap_min_drawing = bounds_estimation_gap_inf_min(k_sequence_drawing + kIndexC);

% diameters_average_drawing = diameters_average(k_sequence_drawing + kIndexC);
% diameters_max_drawing = diameters_max(k_sequence_drawing + kIndexC);
% diameters_min_drawing = diameters_min(k_sequence_drawing + kIndexC);
% 
% bounds_estimation_gap_average_drawing = bounds_estimation_gap_average(k_sequence_drawing + kIndexC);
% bounds_estimation_gap_max_drawing = bounds_estimation_gap_max(k_sequence_drawing + kIndexC);
% bounds_estimation_gap_min_drawing = bounds_estimation_gap_min(k_sequence_drawing + kIndexC);

selected_trial = 7;
selected_n_o_num = 2;

% for selected_trial = 21: 50
%     for selected_n_o_num = 2: 2
        figure,
        subplot(2, 1, 1)
        shadedata=zeros(2,2*length(k_sequence_drawing));
        shadedata(1,1:length(k_sequence_drawing))=k_sequence_drawing;
        shadedata(1,length(k_sequence_drawing)+1:2*length(k_sequence_drawing))=k_sequence_drawing(end:-1:1);
        shadecolor=[100,149,237]/255;
        shadedata(2,1:length(k_sequence_drawing))=diameters_max_drawing;
        reversedata=diameters_min_drawing;
        shadedata(2,length(k_sequence_drawing)+1:2*length(k_sequence_drawing))=reversedata(end:-1:1);
        h=fill(shadedata(1,:)',shadedata(2,:)',shadecolor);
        set(h,'LineStyle','none')
        hold on;
        plot(k_sequence_drawing, diameters_average_drawing, 'LineWidth', 1.2)
        
        plot(k_sequence_drawing, diameters_inf(:, selected_trial, selected_n_o_num), '-.', 'LineWidth', 1.2)
        k_reset = highlights{selected_trial, selected_n_o_num}.k_reset;
        indices_k_reset_draw = find(k_reset == 1);
        plot(k_sequence_drawing(indices_k_reset_draw), diameters_inf(indices_k_reset_draw, selected_trial, selected_n_o_num), '*', 'LineWidth', 1.2)
        
        set(gca,'FontSize',12);
        legend({'Diameter range', 'Averaged diameter', 'One trial', 'Activated Line 5'}, 'NumColumns', 2)
        xlabel('Time Step')
        ylabel('Diameter')
        ylim([0 60])
%         title([num2str(selected_trial) ', ' num2str(selected_n_o_num)])
        grid on;

        subplot(2, 1, 2)
        shadedata=zeros(2,2*length(k_sequence_drawing));
        shadedata(1,1:length(k_sequence_drawing))=k_sequence_drawing;
        shadedata(1,length(k_sequence_drawing)+1:2*length(k_sequence_drawing))=k_sequence_drawing(end:-1:1);
        shadecolor=[100,149,237]/255;
        shadedata(2,1:length(k_sequence_drawing))=bounds_estimation_gap_max_drawing;
        reversedata=bounds_estimation_gap_min_drawing;
        shadedata(2,length(k_sequence_drawing)+1:2*length(k_sequence_drawing))=reversedata(end:-1:1);
        h=fill(shadedata(1,:)',shadedata(2,:)',shadecolor);
        set(h,'LineStyle','none')
        hold on;
        plot(k_sequence_drawing, bounds_estimation_gap_average_drawing, 'LineWidth', 1.2)
        
        plot(k_sequence_drawing, bounds_estimation_gap_inf(:, selected_trial, selected_n_o_num), '-.', 'LineWidth', 1.2)
        plot(k_sequence_drawing(indices_k_reset_draw), bounds_estimation_gap_inf(indices_k_reset_draw, selected_trial, selected_n_o_num), '*', 'LineWidth', 1.2)
        
        set(gca,'FontSize',12);
        legend({'Bound range', 'Averaged bound', 'One trial', 'Activated Line 5'}, 'NumColumns', 2)
        xlabel('Time Step')
        ylabel('Estimation Gap Bound')
        ylim([0 40])
        grid on;
%     end
% end


%%  Component-Wise Intervals
for i = 1: n
    interval_max = zeros(size(k_sequence_drawing));
    interval_min = zeros(size(k_sequence_drawing));
    interval_center = zeros(size(k_sequence_drawing));
    
    for k = k_sequence_drawing
        interval_max(k+kIndexC) = highlights{selected_trial, selected_n_o_num}.IH_posterior_OITCZSMF_real{k+kIndexC}.c(i) + highlights{selected_trial, selected_n_o_num}.IH_posterior_OITCZSMF_real{k+kIndexC}.cwb(i);
        interval_min(k+kIndexC) = highlights{selected_trial, selected_n_o_num}.IH_posterior_OITCZSMF_real{k+kIndexC}.c(i) - highlights{selected_trial, selected_n_o_num}.IH_posterior_OITCZSMF_real{k+kIndexC}.cwb(i);
        interval_center(k+kIndexC) = highlights{selected_trial, selected_n_o_num}.IH_posterior_OITCZSMF_real{k+kIndexC}.c(i);
    end
    
    figure,
    plot(k_sequence_drawing, interval_max, '--', k_sequence_drawing, interval_min, '--', k_sequence_drawing, interval_center, k_sequence_drawing, highlights{selected_trial, selected_n_o_num}.x_sequence(i, :), 'LineWidth', 1.2)
    set(gca,'FontSize',12);
    legend('Max', 'Min', 'Centroid', 'True State')
    xlabel('Time Step')
    if i == 1
        ylabel('x^{(1)}')
    elseif i == 2
        ylabel('x^{(2)}')
    elseif i == 3
        ylabel('x^{(3)}')
    elseif i == 4
        ylabel('x^{(4)}')
    elseif i == 5
        ylabel('x^{(5)}')
    elseif i == 6
        ylabel('x^{(6)}')
    elseif i == 7
        ylabel('x^{(7)}')
    elseif i == 8
        ylabel('x^{(8)}')
    elseif i == 9
        ylabel('x^{(9)}')
    else
        ylabel('x^{(10)}')
    end
        
%     xlim([0 50])
    grid on;
end