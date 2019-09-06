%%%%%
%%%%%
%%%%%   Creation du laplacian 2D
%%%%%
%%%%%

%         Novembre 2016,
%         Jean-Christophe Loiseau
%         loiseau.jc@gmail.com

%-----> Besoin d'écrire la notice explicative.

function D2_2d = laplacian_2D(x, y)

  %--> Nombre de points de maillage dans chaque direction.
  nx = length(x);
  ny = length(y);

  %--> Création de matrices identité.
  Ix = speye(nx);
  Iy = speye(ny);

  %--> Récupère les laplacian 1D
  D2x = laplacian_1D(x);
  D2y = laplacian_1D(y);

  %--> Construit le laplacien bi-dimensionel.
  D2_2d = kron(D2x, Iy) + kron(Ix, D2y);

end
