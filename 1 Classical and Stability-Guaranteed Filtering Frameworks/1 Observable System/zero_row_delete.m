function [G, theta] = zero_row_delete(G, theta)

r = size(G, 1);

index_record_zero = [];

flag = 0;
for i = 1: r
    if norm(G(i, :)) == 0
        index_record_zero = [index_record_zero i];
        if theta(i) < 0
            warning('Infeasible!')
        end
        flag = 1;
    end
end

if flag == 1
    G(index_record_zero, :) = [];
    theta(index_record_zero, :) = [];
end