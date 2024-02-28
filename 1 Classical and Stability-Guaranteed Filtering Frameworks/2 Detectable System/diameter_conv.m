function d = diameter_conv(vertices)
%   Calculate the diameter of a convex set with V-representation
%   The size of vertices is m x n, where m is the number of vertices and n is the dimension of each vertex

m = size(vertices, 1);

d = 0;

for i = 1: m
    for j = 2: m
        if i < j
            temp = norm(vertices(i, :) - vertices(j, :));
            if temp > d
                d = temp;
            end
        end
    end
end