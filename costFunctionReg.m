function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% To start, the cost and gradient will both be unregularized.
[J, grad] = costFunction(theta, X, y);

% Knowing the bias parameter (theta(1)) is not used to regularize,
% we can create a new array which replaces it with zero, which allows
% all but the bias parameter to be regularized in the gradient.
theta_1_eq_0 = [0; theta(2:length(theta))];

% Finally, we can update each output with their regularization factors.
J = J + (lambda/(2*m))*sum(theta_1_eq_0.^2);
grad = grad + (lambda/m)*theta_1_eq_0;

% =============================================================


end
