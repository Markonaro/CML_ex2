function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Calculating the logistic regression (sigmoid) of all examples (X)
% against weights (theta).
ghX = sigmoid(X*theta);

% Computing the cost of the each feature for all examples in set X 
% after being adjusted by weights (theta), and adding a regularization
% term to mitigate the significance of each weight in thteta.
J = (1/m)*sum(-y.*log(ghX)-(1-y).*log(1-ghX)) + ...
    (lambda/(2*m))*sum(theta.^2);

% Factoring in regularization to the remaining weights.
for i = 1:(length(theta))
    if i == 1
        grad = (1/m)*sum((ghX-y).*X);
    else
        grad = (1/m)*sum((ghX-y).*X) + (lambda/m)*theta(i);
    end
end

% =============================================================

end
