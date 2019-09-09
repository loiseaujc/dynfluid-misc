function P = sor(A, omega)

  % --> Diagonal of A.
  D = diag(diag(A));

  % --> Low triangular component of A.
  L = tril(A, k=-1);

  % --> Compute P.
  P = D/omega + L;

end
