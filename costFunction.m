function [J, grad] = costFunction(theta, X, y)
% COSTFUNCTION: Compute cost and gradient for logistic regression.

% Computes both the cost of using theta as the parameters for 
% logistic regression and the gradient of the cost w.r.t. the parameters.

% Number of training examples.
m = length(y);

% Weighted training data transposed by the sigmoid function.
ghX = sigmoid(X*theta);

% The discrepency between what the current theta values predict and what
% the actual values are
J = (1/m)*sum(-y.*log(ghX)-(1-y).*log(1-ghX));

% How severe the error of J is relative to its minimum (J'(theta)).
grad = (1/m)*sum((ghX-y).*X);

end
