function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
%Add an extra column of 1's at the beginning of X as the bias nodes
X = [ones(m, 1) X];
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for i = 1 : m
  z = Theta1 * X(i, :)';
  temp = sigmoid(z);
  a1 = [ones(1, 1); temp]; % Add an row of 1s at the beginning 
  zz = Theta2 * a1;
  a2 = sigmoid(zz);
  
  [x, p(i)] = max(a2);
endfor






% =========================================================================


end
