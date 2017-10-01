function [y_train,y_test]=get_regression_outputs(train_x, train_t, ...
                                                 test_x, degree, lambda)
    X=generate_polynomial_tensor(train_x, degree);
    % Use the derivative of the cost function to build the weights vector
    x_prod = X * X';
    if lambda ~= 0
        x_prod = x_prod + lambda * eye(degree+1);
    end
    weights = linsolve(x_prod, X * train_t);
    % Get the predicted values for the training and test sets
    y_train = X' * weights;
    X_test = generate_polynomial_tensor(test_x,degree);
    y_test = X_test' * weights;
end


function X=generate_polynomial_tensor(x,degree)
    % Creates tensor where column i is the ith data vector    
    numb_elements=size(x,1);
    if numb_elements<degree+1
        x=[x;zeros(degree+1-numb_elements,1)];
    end
    X=flipud(vander(x)'); 
    X=X(1:degree+1,1:numb_elements);
end