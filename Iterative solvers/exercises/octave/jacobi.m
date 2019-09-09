function P = jacobi(A, omega)

  % --> Extraction de la diagonal de A.
  P = diag(diag(A)) / omega;

end
