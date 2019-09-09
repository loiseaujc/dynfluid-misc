function P = ssor(A, omega)

  % --> Diagonal of A.
  D = diag(diag(A));

  % --> Low triangular component of A.
  L = tril(A, k=-1);

  % --> Compute P.
  P = (D/omega + L) * (inv(D)/(2-omega)) * (D/omega + L');

end
