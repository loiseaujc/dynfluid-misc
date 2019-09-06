%%%%%
%%%%%
%%%%%     Imposition des conditions aux limites.
%%%%%
%%%%%

%         Novembre 2016,
%         Jean-Christophe Loiseau
%         loiseau.jc@gmail.com

%-----> Besoin d'écrire la notice explicative.

function f = conditions_limites(f, x, y)

  dx = x(2) - x(1);
  y = y';

  %--> On incorpore les conditions aux limites de Dirichlet non-homogènes
  %    comme un terme de forçage.
  f(:, end) = f(:, end) - y.*(1-y)/dx.^2;

  %--> On transforme notre forçage bidimensionel en un vecteur.
  f = reshape(f, numel(f), 1);

end
