function cZ_prior = cZ_prediction(A, B, cZ_posterior, cZ_w)
%   Prediction step in constrained zonotopic linear set-membership filtering
%   (c) Yirui Cong, created: 4-May-2021, last modified: --

G_posterior = cZ_posterior.G;
c_posterior = cZ_posterior.c;
A_posterior = cZ_posterior.A;
b_posterior = cZ_posterior.b;
cwb_posterior = cZ_posterior.cwb;

G_w = cZ_w.G;
c_w = cZ_w.c;
A_w = cZ_w.A;
b_w = cZ_w.b;
cwb_w = cZ_w.cwb;

ng_posterior = size(G_posterior, 2);
nc_posterior = size(A_posterior, 1);
ng_w = size(G_w, 2);
nc_w = size(A_w, 1);

G_prior = [A * G_posterior, B * G_w];
c_prior = A * c_posterior + B * c_w;
% A_prior = [A_posterior, zeros(nc_posterior, ng_w); zeros(nc_w, ng_posterior), A_w];
% b_prior = [b_posterior; b_w];

if isempty(A_posterior) && isempty(A_w)
    A_prior = [];
    b_prior = [];
elseif isempty(A_posterior)
    A_prior = [zeros(nc_w, ng_posterior), A_w];
    b_prior = b_w;
elseif isempty(A_w)
    A_prior = [A_posterior, zeros(nc_posterior, ng_w)];
    b_prior = b_posterior;
else
    A_prior = [A_posterior, zeros(nc_posterior, ng_w); zeros(nc_w, ng_posterior), A_w];
    b_prior = [b_posterior; b_w];
end

cwb_prior = [cwb_posterior; cwb_w];


% if isempty(A_posterior)
%     if isempty(b_posterior)
% %         A_prior = A_w;
% %         b_prior = b_w;
%         A_prior = [A_posterior, zeros(nc_posterior, ng_w); zeros(nc_w, ng_posterior), A_w];
%         b_prior = [b_posterior; b_w];
%     else
%         error('Incorrect pair (A_posterior, b_posterior)');
%     end
% else
%     A_prior = [A_posterior, zeros(nc_posterior, ng_w); zeros(nc_w, ng_posterior), A_w];
%     b_prior = [b_posterior; b_w];
% end

% [A_prior, b_prior] = zero_row_delete(A_prior, b_prior);
% [G_prior, A_prior] = zero_column_delete(G_prior, A_prior);

cZ_prior = cZ_construct(G_prior, c_prior, A_prior, b_prior, cwb_prior);