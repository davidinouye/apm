function A = randdirichlet(M, N, alpha)
%RANDDIRICHLET Sample from dirichlet where columns are Dirichlet vectors

A = randgamma(M, N, 1, alpha);
A = bsxfun(@rdivide, A, sum(A,1));

end

