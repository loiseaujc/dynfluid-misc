%%%%%
%%%%%
%%%%%     Implementation de la methode de Jacobi.
%%%%%
%%%%%

%         Novembre 2016,
%         Jean-Christophe Loiseau
%         loiseau.jc@gmail.com

%-----> La méthode de Jacobi est un solveur itératif qui peut être utilisé
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
%       la méthode de Jacobi peut s'écrire sous forme matricielle comme:
%
%                 x^(k+1) = inv(D)*(b - R*x^(k))
%
%       où D est la diagonale de la matrice A et R = A-D. Dans MatLab, cette
%       façon d'écrire le problème est beaucoup plus efficace en terme de mémoire
%       et de vitesse d'exécution.

function [sol, iter, converged] = weighted_jacobi(A, b, omega, maxiter, tol)

  %--> Creation de notre approximation initiale.
  x = zeros(size(b));
  converged = false;

  %--> Extraction de la diagonale de A.
  D = diag(A);
  invD = 1./D;

  %--> Boucle de Jacobi.
  for i = 1:maxiter
    %-> Nouvelle approximation.
    x += omega*invD.*(b - A*x);

    %-> Verifie la qualité de notre solution.
    residu = norm(A*x-b)/norm(b);
    if (residu<tol)
      %-> Si notre condition est vérifiée, alors on sort de la boucle.
      converged = true;
      break
    end

    %--> Store the final approximation in to sol.
    sol = x;

    %--> Reports the number of iterations performed.
    iter = i;

end
