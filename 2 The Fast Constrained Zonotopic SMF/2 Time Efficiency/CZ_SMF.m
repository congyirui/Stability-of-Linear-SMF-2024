function cZ_posterior = CZ_SMF(A, B, C, cZ_in, y, k, cZ_w, cZ_v, reduction_method, nc, od)
%   CZ-SMF
%   Toolbox CORA 2024 is needed.
%
%   (c) Yirui Cong, created: 12-Oct-2021, last modified: 21-Feb-2024


if k == 0
    cZ_prior = cZ_in;
else
    cZ_prior = cZ_prediction(A, B, cZ_in, cZ_w);
end

cZ_posterior = cZ_update(C, eye(size(C, 1)), y, cZ_prior, cZ_v, 0);

cZ_posterior_CORA = conZonotope(cZ_posterior.c, cZ_posterior.G * diag(cZ_posterior.cwb), cZ_posterior.A * diag(cZ_posterior.cwb), cZ_posterior.b);

cZ_posterior_CORA = rescale(cZ_posterior_CORA, 'iter');
cZ_posterior_CORA = reduceConstraints(cZ_posterior_CORA, nc);
cZ_posterior_CORA = reduce(cZ_posterior_CORA, reduction_method, od);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'girard', od);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'combastel', od);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'scott', od);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'pca', od);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'constOpt', od);

% cZ_posterior = cZ_construct(cZ_posterior_CORA.Z(:, 2: end), cZ_posterior_CORA.Z(:, 1), cZ_posterior_CORA.A, cZ_posterior_CORA.b, ones(size(cZ_posterior_CORA.A, 2), 1));
cZ_posterior = cZ_construct(cZ_posterior_CORA.G, cZ_posterior_CORA.c, cZ_posterior_CORA.A, cZ_posterior_CORA.b, ones(size(cZ_posterior_CORA.A, 2), 1));