function [train_err,validation_err,test_err,y_train,y_test]= perform_cross_validation(train_x, train_t, ...
                                                                                      test_x, test_t, ...
                                                                                      degree, lambdas, k, ignoreBias)
    % Construct the data structure to store the results
    train_err=zeros([length(lambdas),k]);
    validation_err=zeros([length(lambdas),k]);
    test_err=zeros(length(lambdas),1);
    
    % Make the cross validation index splits
    cvFolds = crossvalidind_zayd(length(train_x),k);
        
    % Test all the lambdas
    for cnt_lam = 1:length(lambdas)
        lambda = lambdas(cnt_lam);
        for i = 1:k
            % build the test and validation sets
            v_ids = (cvFolds == i); % Row Ids for the validation set
            t_ids = ~v_ids;         % Row IDs for the TRAINING set    
            % Perform regression then store the outputs
            [y_train,y_v]=get_regression_outputs(train_x(t_ids,:),train_t(t_ids,:), ...
                                                 train_x(v_ids,:), degree, lambda, ignoreBias);
            train_err(cnt_lam,i)=rms(y_train - train_t(t_ids,:));
            validation_err(cnt_lam,i)=rms(y_v - train_t(v_ids,:)); 
        end
        % Use all the data to build the test result
        [y_train,y_test]=get_regression_outputs(train_x, train_t, test_x, ...
                                          degree, lambda, ignoreBias);
        test_err(cnt_lam)=rms(y_test - test_t);
    end
end


% Uniform or near uniform random split for cross validation.
function cv_folds = crossvalidind_zayd(n,k)
    id_list = randperm(n);
    cv_folds = zeros(n,1);
    for i = 1:n
        cv_folds(id_list(i)) = mod(i,k) + 1;
    end
end 
