function [G_ROIT, theta_ROIT] = OIT(A, B, C, ySequence, k, delta, G_w, theta_w, G_v, theta_v, mu)
%   Reduced Observation-Information Tower
%   (c) Yirui Cong, created: 28-Jan-2020, last modified: 07-Feb-2020

n = size(A, 1);
p = size(B, 2);
rr = size(G_w, 1);

G_ROIT = [];
theta_ROIT = [];

% [G_Bw, theta_Bw] = sop2p([eye(n); -eye(n); zeros(rr, n)], [-B; B; G_w], [zeros(n, 1); zeros(n, 1); theta_w]);
% [G_Bw, theta_Bw] = noredund(G_Bw, theta_Bw);
% rr_Bw = size(G_Bw, 1);

kIndexC = 1; % A compensator for the index 0 in matlab: for example, y[0] in matlab is y(1) = y(0 + k_indexC).

for i = k-delta: k
    if i < k
        G_Oki_ = [-G_v*C/A^(k-i); zeros((k-i)*rr, n)];
        theta_Oki_ = [theta_v - G_v*ySequence(:, i+kIndexC); kron(ones(k-i, 1), theta_w)];
        H_Oki_ = [];
        for r = i: k-1
            H_Oki_ = [H_Oki_, G_v*C/(A^(r+1-i))*B];
        end
        H_Oki_ = [H_Oki_; kron(eye(k-i), G_w)];
%         H_Oki_ = zeros((k-i+1)*rr, (k-i)*p);
%         for r = i: k-1
%             H_Oki_(1: rr, (r-i)*p+1: (r-i+1)*p) = G_v*C/A^(r+1-i)*B;
%             H_Oki_((r-i+1)*rr+1: (r-i+2)*rr, (r-i)*p+1: (r-i+1)*p) = G_w;
%         end
        
        [G_Oki, theta_Oki] = sop2p(G_Oki_, H_Oki_, theta_Oki_, 0);
        
        G_ROIT = [G_ROIT; G_Oki];
        theta_ROIT = [theta_ROIT; theta_Oki];
    else
        G_Okk = -G_v * C;
        theta_Okk = theta_v - G_v * ySequence(:, k+kIndexC);
        
        G_ROIT = [G_ROIT; G_Okk];
        theta_ROIT = [theta_ROIT; theta_Okk];
    end
end

if k >= delta
    [G_ROIT, theta_ROIT] = noredund(G_ROIT, theta_ROIT);
end