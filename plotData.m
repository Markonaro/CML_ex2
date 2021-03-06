function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure;

hold on;

% Find Indices of Positive and Negative Examples
p = find(y == 1);
n = find(y == 0);

% Plot Examples
plot(X(p, 1), X(p, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(n, 1), X(n, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);

hold off;

end
