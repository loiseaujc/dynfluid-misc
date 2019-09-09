function P = gauss_seidel(A, omega)

  %--> Extraction de L, la partie triangulaire inférieure de A.
  P = tril(A, k=0);

end
