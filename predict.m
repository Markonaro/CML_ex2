function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% =========================================================================

% Weighting all entries by derived theta paramters, then transposing to
% their probability (between 0 and 1) via the sigmoid function.
ghX = sigmoid(X*theta);

% For each training example, determine predicted outcome. All values less
% than 0.5 should return 0, whilst values >= 0.5 return 1.
p = round(ghX);

% =========================================================================


end
