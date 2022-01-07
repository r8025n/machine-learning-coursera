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
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
z = X * theta;
h = sigmoid(z);
term1 = -1 * y .* log(h);
term2 = (1 - y) .* log(1 - h);
term = term1 - term2;
temp = ones(1, size(term, 1));
reg_term = (lambda / (2*m)) .* (ones(1, size(theta) - 1) * (theta(2 : size(X, 2), :) .* theta(2 : size(X, 2), :)));
##size(reg_term)
J = (temp * term) / m;
##size(J)
J = J + reg_term;
##size(J)

temp = X' * (h - y);
reg_term2 = (lambda / m) * theta(2 : size(X, 2), :);
grad = temp /m;
grad(2 : size(X, 2), :) = grad(2 : size(X, 2), :) + reg_term2;





% =============================================================

end
