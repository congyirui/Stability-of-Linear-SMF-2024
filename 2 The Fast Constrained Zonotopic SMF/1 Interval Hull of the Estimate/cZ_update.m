function cZ_posterior = cZ_update(C, D, y, cZ_prior, cZ_v, flag)
%   Update step in constrained zonotopic linear set-membership filtering
%   (c) Yirui Cong, created: 4-May-2021, last modified: --

G_prior = cZ_prior.G;
c_prior = cZ_prior.c;
A_prior = cZ_prior.A;
b_prior = cZ_prior.b;
cwb_prior = cZ_prior.cwb;

G_v = cZ_v.G;
c_v = cZ_v.c;
A_v = cZ_v.A;
b_v = cZ_v.b;
cwb_v = cZ_v.cwb;

n = size(C, 2);
ng_prior = size(G_prior, 2);
nc_prior = size(A_prior, 1);
ng_v = size(G_v, 2);
nc_v = size(A_v, 1);

G_posterior = [G_prior, zeros(n, ng_v)];
c_posterior = c_prior;

if isempty(A_prior) && isempty(A_v)
    A_posterior = [C * G_prior, D * G_v];
    b_posterior = y - D * c_v - C * c_prior;
elseif isempty(A_prior)
    A_posterior = [zeros(nc_v, ng_prior), A_v; C * G_prior, D * G_v];
    b_posterior = [b_v; y - D * c_v - C * c_prior];
elseif isempty(A_v)
    A_posterior = [A_prior, zeros(nc_prior, ng_v); C * G_prior, D * G_v];
    b_posterior = [b_prior; y - D * c_v - C * c_prior];
else
    A_posterior = [A_prior, zeros(nc_prior, ng_v); zeros(nc_v, ng_prior), A_v; C * G_prior, D * G_v];
    b_posterior = [b_prior; b_v; y - D * c_v - C * c_prior];
end

cwb_posterior = [cwb_prior; cwb_v];

cZ_posterior = cZ_construct(G_posterior, c_posterior, A_posterior, b_posterior, cwb_posterior);

if flag == 1
    cZ_posterior = cZ_column_scaling(cZ_posterior);
end