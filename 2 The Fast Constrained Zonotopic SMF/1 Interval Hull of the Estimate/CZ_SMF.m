function cZ_posterior = CZ_SMF(A, B, C, cZ_in, y, k, cZ_w, cZ_v, reduction_method, ro, B_u, u)
%   CZ-SMF (existing methods)
%   Toolbox CORA 2020 is needed.
%
%   (c) Yirui Cong, created: 12-Oct-2021, last modified: 05-Oct-2022


if k == 0
    cZ_prior = cZ_in;
else
    cZ_prior = cZ_prediction(A, B, cZ_in, cZ_w, B_u, u);
end

cZ_posterior = cZ_update(C, eye(size(C, 1)), y, cZ_prior, cZ_v, 0);

cZ_posterior_CORA = conZonotope(cZ_posterior.c, cZ_posterior.G, cZ_posterior.A, cZ_posterior.b);

cZ_posterior_CORA = rescale(cZ_posterior_CORA, 'iter');
cZ_posterior_CORA = reduce(cZ_posterior_CORA, reduction_method, ro);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'girard', ro);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'combastel', ro);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'scott', ro);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'pca', ro);
% cZ_posterior_CORA = reduce(cZ_posterior_CORA, 'constOpt', ro);

cZ_posterior = cZ_construct(cZ_posterior_CORA.Z(:, 2: end), cZ_posterior_CORA.Z(:, 1), cZ_posterior_CORA.A, cZ_posterior_CORA.b, ones(size(cZ_posterior_CORA.A, 2), 1));