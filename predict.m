function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% Number of training examples.
m = size(X, 1);

% Initialize p to have equivalent size to the # of training examples.
p = zeros(m, 1);

% Weighting all entries by derived theta paramters, then transposing to
% their probability via the sigmoid function.
ghX = sigmoid(X*theta);

% For each training example, determine predicted outcome.
for i=1:m
   if (ghX(i) < 0.5)
       p(i) = 0;
   else
       p(i) = 1;
   end
end



% =========================================================================


end
