%%%%%
%%%%%
%%%%%   Creation du laplacian 2D
%%%%%
%%%%%

%         Novembre 2016,
%         Jean-Christophe Loiseau
%         loiseau.jc@gmail.com

%-----> Besoin d'écrire la notice explicative.

function D2_3d = laplacian_3D(x, y, z)

  %--> Nombre de points de maillage dans chaque direction.
  nx = length(x);
  ny = length(y);
  nz = length(z);

  %--> Création de matrices identité.
  Ix = speye(nx);
  Iy = speye(ny);
  Iz = speye(nz);

  %--> Récupère les laplacian 1D
  D2x = laplacian_1D(x);
  D2y = laplacian_1D(y);
  D2z = laplacian_1D(z);

  %--> Construit le laplacien bi-dimensionel.
  D2_3d = kron(D2x, Iy, Iz) + kron(Ix, D2y, Iz) + kron(Ix, Iy, D2z);

end
