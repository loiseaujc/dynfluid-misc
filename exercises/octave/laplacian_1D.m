%%%%%
%%%%%
%%%%%     Construction du laplacian 1D
%%%%%
%%%%%

%         Novembre 2016,
%         Jean-Christophe Loiseau
%         loiseau.jc@gmail.com

%-----> Cette fonction permet de construire la matrice de dérivée seconde
%       obtenue via un schéma aux différences finies centrées d'ordre 2.
%       Pour ce faire, l'opérateur de dérivée seconde est approximé comme :
%
%                 df/dx = (f(i+1) - 2*f(i) + f(i-1))/dx^2
%
%       À noter que dans la fonction qui suit, on présuppose des conditions
%       aux limites de Dirichlet homogènes, i.e. f = 0 aux bords du domaine.
%
%       Note: Voir la fonction conditions_limites pour l'imposition de
%       conditions aux limites de Dirichlet non-homogènes.

function D2 = laplacian_1D(x)

  %--> Nombre de points du maillage x.
  n = length(x);

  %--> Calcul Delta x.
  dx = x(2) - x(1);

  % --> Création du laplacien 1D.
  d2 = [1*ones(n, 1), -2*ones(n, 1), 1*ones(n, 1)];

  D2 = spdiags(d2, [-1, 0, 1], n, n) / dx^2;

end
