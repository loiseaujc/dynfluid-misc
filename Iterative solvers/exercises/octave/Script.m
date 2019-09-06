%#####
%#####
%#####      Resolution de l'équation de la chaleur par des méthodes itératives.
%#####
%#####

%         Novembre 2016,
%         Jean-Christophe Loiseau
%         loiseau.jc@gmail.com

clear
close all

%--> Création du maillage.
nx = 128;                          % Nombre de points dans la direction x.
ny = 128;                          % Nombre de points dans la direction y.

x = linspace(0, 1, nx+2);         % Création du maillage dans la direction x.
y = linspace(0, 1, ny+2);         % Création du maillage dans la direction y.

x = x(2:end-1);                   % On retire les points sur les bords. Ce ne
y = y(2:end-1);                   % sont pas de vrais degrés de liberté du problème.

[X, Y] = meshgrid(x, y);          % Création du maillage 2D.

%--> Solution analytique.
T_ref = Y.*(1-Y).*X.^3;

% %--> Plot le champ de temperature.
% figure(1)
% surface(X, Y, T_ref, 'EdgeColor', 'None', 'FaceColor', 'interp');
% title('Solution analytique T(x,y)');
% xlabel('x');
% ylabel('y');
%
% %--> Plot le profil de température le long de x = 0.5 (i.e. le 25ème point sur 49)
% figure(2)
% plot(y, T_ref(:, 25));

%--> Creation du laplacian 2D.
D2 = laplacian_2D(x, y);

%--> Creation du terme de forcage.
F = 6*X.*Y.*(1-Y) - 2*X.^3;

% %--> Plot le terme de forcage.
% figure(3)
% surface(X, Y, F, 'EdgeColor', 'None', 'FaceColor', 'interp');
% title('Forcage f(x,y)');
% xlabel('x');
% ylabel('y');

%--> Ajout des conditions aux limites.
F_bc = conditions_limites(F, x, y);

D2 = -D2;
F_bc = -F_bc;



%--------------------------------------------------------
%-----                                              -----
%-----     Résolution avec la méthode de Jacobi     -----
%-----                                              -----
%--------------------------------------------------------

printf('Résolution de l équation de la chaleur via une méthode itérative \n')
printf('---------------------------------------------------------------- \n')
maxiter = 1e5;                              % Nombre d'itérations maximum.
tol = 1e-8;                                   % Tolérance pour le résidu.
[sol, iter, converged] = jacobi(D2, F_bc, maxiter, tol);

%--> Affichage à l'écran de la convergence ou non de la méthode de Jacobi.
if (converged == true)
    printf('  --> La méthode de Jacobi a convergé en %i itérations.', iter);
else
    printf('  --> La méthode de Jacobi n a pas réussi à converger.');
end
printf('\n')

% %--> Reformatage de la solution sous la forme d'un champ 2D.
% T_jacobi = reshape(sol, ny, nx);

% %--> Plot la solution obtenue avec la méthode de Jacobi.
% figure(4)
% surface(X, Y, T_jacobi, 'EdgeColor', 'None', 'FaceColor', 'interp');
% title('Méthode de Jacobi');
% xlabel('x');
% ylabel('y');





%----------------------------------------------------------------
%-----                                                      -----
%-----     Résolution avec la méthode de Jacobi pondéré     -----
%-----                                                      -----
%----------------------------------------------------------------

omega = 1.66;
[sol, iter, converged] = weighted_jacobi(D2, F_bc, omega, maxiter, tol);

%--> Affichage à l'écran de la convergence ou non de la méthode de Gauss-Seidel.
if (converged == true)
    printf('  --> La méthode de Jacobi pondéré a convergé en %i itérations.', iter);
else
    printf('  --> La méthode de Jacobi pondéré n a pas réussi à converger.');
end
printf('\n')


%
% %--> Affichage à l'écran de la convergence ou non de la méthode de Jacobi.
% if (converged == true)
%     printf('  --> La méthode de Jacobi a convergé en %i itérations.', iter);
% else
%     printf('  --> La méthode de Jacobi n a pas réussi à converger.');
% end
% printf('\n')
%
% %--> Reformatage de la solution sous la forme d'un champ 2D.
% T_jacobi = reshape(sol, ny, nx);

% %--> Plot la solution obtenue avec la méthode de Jacobi.
% figure(4)
% surface(X, Y, T_jacobi, 'EdgeColor', 'None', 'FaceColor', 'interp');
% title('Méthode de Jacobi');
% xlabel('x');
% ylabel('y');





%--------------------------------------------------------------
%-----                                                    -----
%-----     Résolution avec la méthode de Gauss-Seidel     -----
%-----                                                    -----
%--------------------------------------------------------------

[sol, iter, converged] = gauss_seidel(D2, F_bc, maxiter, tol);

%--> Affichage à l'écran de la convergence ou non de la méthode de Gauss-Seidel.
if (converged == true)
    printf('  --> La méthode de Gauss-Seidel a convergé en %i itérations.', iter);
else
    printf('  --> La méthode de Gauss-Seidel n a pas réussi à converger.');
end
printf('\n')

%--> Reformatage de la solution sous la forme d'un champ 2D.
% T_gs = reshape(sol, ny, nx);

% %--> Plot la solution obtenue avec la méthode de Gauss-Seidel.
% figure(5)
% surface(X, Y, T_gs, 'EdgeColor', 'None', 'FaceColor', 'interp');
% title('Méthode de Gauss-Seidel');
% xlabel('x');
% ylabel('y');




%---------------------------------------------------------------------------------
%-----                                                                       -----
%-----     Résolution avec la méthode de sur-relaxation succesives (SOR)     -----
%-----                                                                       -----
%---------------------------------------------------------------------------------

omega = 1.89;     % Poids pour la sur-relaxation.
[sol, iter, converged] = successive_over_relaxation(D2, F_bc, omega, maxiter, tol);

%--> Affichage à l'écran de la convergence ou non de la méthode de Gauss-Seidel.
if (converged == true)
    printf('  --> La méthode SOR a convergé en %i itérations.', iter);
else
    printf('  --> La méthode SOR n a pas réussi à converger.');
end
printf('\n')

%--> Reformatage de la solution sous la forme d'un champ 2D.
% T_sor = reshape(sol, ny, nx);

%--> Plot la solution obtenue avec la méthode de Gauss-Seidel.
% figure(6)
% surface(X, Y, T_sor, 'EdgeColor', 'None', 'FaceColor', 'interp');
% title('Méthode SOR');
% xlabel('x');
% ylabel('y');
