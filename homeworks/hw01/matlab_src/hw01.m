function hw01(degree, lambdas, k)
   train_data=importdata("train.txt");
   test_data=importdata("test.txt");

   for test_lambda=lambdas
       compare_train_test_single_lambda(train_data, test_data, degree, test_lambda)
   end
    
   sweep_lambda(train_data, test_data, degree, lambdas, k)                                                  
end


function compare_train_test_single_lambda(train_data, test_data, degree, lambda)

   [~,~,~,y_train,y_test]=perform_cross_validation(train_data(:,1),train_data(:,2), ...
                                                   test_data(:,1),test_data(:,2), ...
                                                   degree, lambda, 0);
    % Plot the scatters and configure how they will appear
    test_dot_size = 5;
    yellowish = [.8,.6,0];
    scatter(test_data(:,1),test_data(:,2),test_dot_size,yellowish)
    hold on; % Allow multiple plots simultaneously
    dark_blue = [0,0.2,0.8];
    scatter(test_data(:,1),y_test,test_dot_size,dark_blue)
    train_dot_size = 50;
    scatter(train_data(:,1),train_data(:,2),train_dot_size,'black','filled')
    scatter(train_data(:,1),y_train,train_dot_size,'red','filled')
    hold off;
    legend('Test Target','Test Predicted','Train Target','Train Predicted');
    
    % Format the plot
    xlabel('X');
    ylabel('Target');
    title(['Learner Output for ' int2str(degree) '-Polynomial with \lambda=' num2str(lambda,'%1.2f')]);
    filename = ['img/test_train_compare_d=' int2str(degree) '_lambda=' num2str(lambda,'%1.2f') '.eps' ];
    print(filename,'-deps');
end


function sweep_lambda(train_data, test_data, degree, lambdas, k)
   % Get the cross validation errors
   [train_err,valid_err,test_err,~,~]=perform_cross_validation(train_data(:,1),train_data(:,2), ...
                                                               test_data(:,1),test_data(:,2), ...
                                                               degree, lambdas, k);
   
   % Get the average errors
   train_avg_err = [mean(train_err,2), var(train_err')']; %#ok<UDIM>
   valid_avg_err = [mean(valid_err,2), var(valid_err')']; %#ok<UDIM>
   
   % Plot the results
   plot_lambda_sweep(degree, lambdas, train_avg_err, valid_avg_err, test_err);
end


function plot_lambda_sweep(degree, lambdas, train_avg_errs, valid_avg_errs, test_errs)
    errorbar(lambdas,train_avg_errs(:,1),train_avg_errs(:,2));
    hold on; % Allow multiple plots simultaneously
    errorbar(lambdas,valid_avg_errs(:,1),valid_avg_errs(:,2));
    plot(lambdas, test_errs);
    hold off;
    legend('Train','Validation','Test');
    
    % Format the plot
    xlabel('\lambda');
    ylabel('RMS Error');
    title(['Effect of \lambda on the Learning Errors for a ' int2str(degree) '-Polynomial']);
    set(gca,'XScale','log') % Log scale
    set(gca,'YScale','log') % Log scale
    print -deps img/lambda_sweep;
end