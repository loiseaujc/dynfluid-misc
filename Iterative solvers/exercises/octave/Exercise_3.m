clear
close all

%--> Création du maillage.
nx = 32;                          % Nombre de points dans la direction x.
ny = nx;
nz = nx;

x = linspace(0, 1, nx+2);         % Création du maillage dans la direction x.
y = linspace(0, 1, ny+2);         % Création du maillage dans la direction y.
z = linspace(0, 1, nz+2);         % Création du maillage dans la direction y.

x = x(2:end-1);                   % On retire les points sur les bords. Ce ne
y = y(2:end-1);                   % sont pas de vrais degrés de liberté du problème.
z = z(2:end-1);                   % sont pas de vrais degrés de liberté du problème.

[X, Y, Z] = meshgrid(x, y, z);          % Création du maillage 2D.

%--> Creation du laplacian 2D.
A = laplacian_3D(x, y, z);

%--> Creation du terme de forcage.
b = ones(nx*ny*nz, 1);

%--------------------------------------------------------
%-----                                              -----
%-----     Résolution avec la méthode de Jacobi     -----
%-----                                              -----
%--------------------------------------------------------

printf('Résolution de l équation de la chaleur via une méthode itérative \n')
printf('---------------------------------------------------------------- \n')
opts.maxiter = 1e6;                              % Nombre d'itérations maximum.
opts.tol = 1e-10;                                   % Tolérance pour le résidu.
opts.omega = 1;

[sol, iter, converged, residu] = iterative_solver(A, b, @jacobi, opts);

figure(1)
loglog(residu);
hold on

%--> Affichage à l'écran de la convergence ou non de la méthode de Jacobi.
if (converged == true)
    printf('  --> La méthode de Jacobi a convergé en %i itérations.', iter);
else
    printf('  --> La méthode de Jacobi n a pas réussi à converger.');
end
printf('\n')





%--------------------------------------------------------------
%-----                                                    -----
%-----     Résolution avec la méthode de Gauss-Seidel     -----
%-----                                                    -----
%--------------------------------------------------------------

opts.omega = 1;
[sol, iter, converged, residu] = iterative_solver(A, b, @gauss_seidel, opts);
plot(residu);
hold on;

%--> Affichage à l'écran de la convergence ou non de la méthode de Gauss-Seidel.
if (converged == true)
    printf('  --> La méthode de Gauss-Seidel a convergé en %i itérations.', iter);
else
    printf('  --> La méthode de Gauss-Seidel n a pas réussi à converger.');
end
printf('\n')




%--------------------------------------------------
%-----                                        -----
%-----     Résolution avec la méthode SOR     -----
%-----                                        -----
%--------------------------------------------------

opts.omega = 1.9;
[sol, iter, converged, residu] = iterative_solver(A, b, @sor, opts);
plot(residu);
hold on;

%--> Affichage à l'écran de la convergence ou non de la méthode de Gauss-Seidel.
if (converged == true)
    printf('  --> La méthode SOR a convergé en %i itérations.', iter);
else
    printf('  --> La méthode SOR n a pas réussi à converger.');
end
printf('\n')




%---------------------------------------------------
%-----                                         -----
%-----     Résolution avec la méthode SSOR     -----
%-----                                         -----
%---------------------------------------------------

opts.omega = 1.9;
[sol, iter, converged, residu] = iterative_solver(A, b, @ssor, opts);
plot(residu);
hold on;

%--> Affichage à l'écran de la convergence ou non de la méthode de Gauss-Seidel.
if (converged == true)
    printf('  --> La méthode SSOR a convergé en %i itérations.', iter);
else
    printf('  --> La méthode SSOR n a pas réussi à converger.');
end
printf('\n')

legend("Jacobi", "Gauss-Seidel", "SOR", "SSOR");
