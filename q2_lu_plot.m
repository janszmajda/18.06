% Q2 LU timing plot (MATLAB/Octave)
% Run pset-1 first to generate q2_lu_timing.csv

data = readmatrix('q2_lu_timing.csv');
n = data(:, 1);
t = data(:, 2);

% Fit T(n) = C * n^alpha using log-log regression
p = polyfit(log(n), log(t), 1);
alpha = p(1);
C = exp(p(2));

tfit = C * n.^alpha;

figure('Color', 'w');
loglog(n, t, 'o', 'MarkerSize', 6, 'LineWidth', 1.2);
hold on;
loglog(n, tfit, '-', 'LineWidth', 1.5);
grid on;
xlabel('n');
ylabel('Time (s)');
title(sprintf('LU timing: T(n) = %.2e n^{%.3f}', C, alpha));
legend('Measured', 'Fit', 'Location', 'northwest');

fprintf('alpha = %.6f\n', alpha);
fprintf('C = %.6e\n', C);
