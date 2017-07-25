function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0; % cost, or how far from best fit we are
grad = zeros(size(theta)); % gradient, rate at which we're approaching min.

% Calculating the logistic regression (sigmoid) of all examples (X)
% against weights (theta).
ghX = sigmoid(X*theta);

% Computing the cost of the each feature for all examples in set X 
% after being adjusted by weights (theta), and adding a regularization
% term to mitigate the significance of each weight in thteta.
J = (1/m)*sum(-y.*log(ghX)-(1-y).*log(1-ghX)) + ...
    (lambda/(2*m))*sum(theta(2:end).^2);

% Getting the "pure" gradient of each term sans regularization
grad = ((1/m)*sum((ghX-y).*X))';

% Knowing the first element of grad doesn't need to be regularized,
% we can create a new array which "adds zero" to grad(1) whilst
% regularizing every subsequent gradient value.
theta_1_eq_0 = [0; theta(2:length(theta))];
grad = grad + (lambda/m)*theta_1_eq_0;

% =============================================================


end
