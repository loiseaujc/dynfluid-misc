function [x, iter, converged, residu] = iterative_solver(A, b, solver, opts)

  % --> Create initial guess.
  x = zeros(size(b));
  converged = false;

  % --> Residual array.
  residu = zeros(opts.maxiter, 1);

  % --> Get the iteration matrix.
  P = solver(A, opts.omega);

  % --> Iterative solver.
  for iter = 1:opts.maxiter
    % --> New approximation.
    x += P\(b - A*x);

    % --> Compute residual.
    residu(iter) = norm(A*x - b) / norm(b);
    if residu(iter) < opts.tol
      converged = true;
      break
    end

    % --> Misc.
    residu = residu(1:iter);

end
