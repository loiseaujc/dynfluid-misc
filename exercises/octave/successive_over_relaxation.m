%%%%%
%%%%%
%%%%%     Implémentation de la methode SOR.
%%%%%
%%%%%

%         Novembre 2016,
%         Jean-Christophe Loiseau
%         loiseau.jc@gmail.com

%-----> La méthode SOR est un solveur itératif qui peut être utilisé
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
%       la méthode SOR peut s'écrire sous forme matricielle comme:
%
%                 x^(k+1) = x^(k) + inv(P)*(b - A*x^(k))
%
%       où P = D/omega + L, avec L la partie triangulaire strictement inférieure
%       de la matrice A et omega le poids de relaxation choisi.

function [sol, iter, converged] = successive_over_relaxation(A, b, omega, maxiter, tol)

  %--> Création de notre approximation initiale.
  x = zeros(size(b));
  converged = false;

  %--> Extraction de la diagonale de A.
  D = diag(diag(A));

  %--> Extraction de L, la partie triangulaire strictement inférieure de A.
  L = tril(A, k=-1);

  %--> Extraction de U, la partie triangulaire strictement supérieure de A.
  U = triu(A, k=1);

  %--> Calcul de P = L + D/omega.
  P = D/omega + L;

  %--> Boucle SOR.
  for i = 1:maxiter
    %-> Nouvelle approximation.
    x += P\(b - A*x);            %Note: Inverser directement la matrice P serait
                                 %      trop coûteux et inutile. Matlab gère très
                                 %      bien la résolution via une méthode dite de
                                 %      'forward substitution', P étant triangulaire
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
