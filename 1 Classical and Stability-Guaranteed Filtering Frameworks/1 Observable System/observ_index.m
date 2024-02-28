function mu = observ_index(A, C)
%   Returns the observability index, given that (A, C) is observable.
%   (c) Yirui Cong, created: 05-Feb-2020

obsvMatrix = obsv(A, C);
[m, n] = size(C);

if rank(obsvMatrix) < n
    error('The pair (A, C) is not observable!')
end

for mu = 1: n
    if rank(obsvMatrix(1: mu*m, :)) == n
        break;
    end
end