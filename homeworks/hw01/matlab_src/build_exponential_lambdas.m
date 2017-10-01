function lambdas=build_exponential_lambdas(pow_val, max_lambda)
    lambdas = 0;
    i = -10;
    % Increase by a power of the input until the max value is reached.
    while pow_val ^ i < max_lambda
        lambdas = [lambdas pow_val ^ i];
        i = i + 1;
    end 
end