%%%%%
%%%%%
%%%%%     Implémentation de la methode de Gauss-Seidel.
%%%%%
%%%%%

%         Novembre 2016,
%         Jean-Christophe Loiseau
%         loiseau.jc@gmail.com

%-----> La méthode de Gauss-Seidel est un solveur itératif qui peut être utilisé
%       sous certaines conditions:
%       - la matrice est à diagonale dominante.
%       - la diagonale ne contient aucun terme nul.
%
%       Dans cette routine, l'implémentation de la méthode peut apparaître
%       légèrement différente de ce que vous avez vu en cours. Il s'agit
%       pourtant exactement de la même chose. En effet, pour résoudre un problème
%       du type
%
%                 A*x = b
%
%       la méthode de Gauss-Seidel peut s'écrire sous forme matricielle comme:
%
%                 x^(k+1) = inv(L)*(b - U*x^(k))
%
%       où L est la partie triangulaire inférieure de la matrice A et U sa partie
%       triangulaire strictement supérieure. Dans MatLab, cette façon d'écrire
%       le problème est beaucoup plus efficace en terme de mémoire et de vitesse
%       d'exécution.

function [sol, iter, converged] = gauss_seidel(A, b, maxiter, tol)

  %--> Création de notre approximation initiale.
  x = zeros(size(b));
  converged = false;

  %--> Extraction de L, la partie triangulaire inférieure de A.
  L = tril(A, k=0);

  %--> Extraction de U, la partie triangulaire strictement supérieure de A.
  U = triu(A, k=1);

  %--> Boucle de Gauss-Seidel.
  for i = 1:maxiter
    %-> Nouvelle approximation.
    x = L\(b - U*x);          %Note: Inverser directement la matrice L serait
                              %      trop coûteux et inutile. Matlab gère très
                              %      bien la résolution via une méthode dite de
                              %      'forward substitution', L étant triangulaire
                              %      inférieure.

    %-> Vérifie la qualité de notre solution.
    residu = norm(A*x-b)/norm(b);
    if (residu<tol)
      %-> Si notre condition est vérifiée, alors on sort de la boucle.
      converged = true;
      break
    end

    %--> Store the final approximation into sol.
    sol = x;

    %--> Reports the number of iterations performed.
    iter = i;

end
