function hw01(degree, lambdas, k)
   train_data=importdata("train.txt");
   test_data=importdata("test.txt");

   for test_lambda=lambdas
       compare_train_test_single_lambda(train_data, test_data, degree, test_lambda)
   end
   
   % K fold
   sweep_lambda(train_data, test_data, degree, lambdas, k)
   % Leave One Out
   sweep_lambda(train_data, test_data, degree, lambdas, size(train_data,1))
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
    leg = legend('Test Target','Test Predicted','Train Target','Train Predicted');
    set(leg,'Location','Best')  % Prevent legend overlap with the data
    
    % Format the plot
    xlabel('X');
    ylabel('Target');
    title(['Learner Output for ' int2str(degree) ...
           '-Polynomial with \lambda=' generate_lambda_string(lambda)]);
    set(gcf, 'Color', 'w'); % Make the background white
    
    % Construct the graph without compressing the axes
    file_folder = 'img/';
    file_name = ['test_train_compare_d=' int2str(degree) ...
                 '_lambda=' generate_lambda_string(lambda)];
    file_extension = '.pdf';
    full_file_path = [ file_folder file_name file_extension ];
    export_fig(full_file_path)
    
    % Allow for smart y-axis so the extremes do not dominate.
    ymin_compressed = 1; 
    ymax_compressed = 5;
    if min(y_test) < ymin_compressed || max(y_test) > ymax_compressed
        filename_comp = [file_name '_compressed'];
        full_file_path = [ file_folder filename_comp file_extension ];
        ylim([ymin_compressed ymax_compressed])
        export_fig(full_file_path)
    end
    
end


% Used to make cleaner printing of the lambda value in the
% title of graphs.
function lamda_str=generate_lambda_string(lambda)
    if lambda>=1
        precision = ceil(log10(lambda));
    else
        precision = -1 * floor(log10(lambda)) + 1;
    end
    lamda_str = num2str(lambda,precision);
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
   if (k~=size(train_data,1)); folds=k; else; folds=NaN; end
   plot_lambda_sweep(degree, lambdas, train_avg_err, valid_avg_err, test_err, folds);
end


function plot_lambda_sweep(degree, lambdas, train_avg_errs, valid_avg_errs, test_errs, k)
    errorbar(lambdas,train_avg_errs(:,1),train_avg_errs(:,2));
    hold on; % Allow multiple plots simultaneously
    errorbar(lambdas,valid_avg_errs(:,1),valid_avg_errs(:,2));
    plot(lambdas, test_errs);
    hold off;
    leg = legend('Train','Validation','Test');
    set(leg,'Location','Best')  % Prevent legend overlap with the data
    
    % Document k-Fold versus Leave One Out
    if isnan(k) == 0 % isnan returns 0 when is not NaN
        filename_k = [ 'k=' int2str(k) ];
        title_str = [int2str(k) '-Fold'];
    else
        filename_k = 'leave_one_out';
        title_str = 'Leave-One-Out';
    end
    
    % Format the plot
    xlim([0 max(lambdas)]);
    xlabel('\lambda');
    ylabel('RMS Error');
    title(['Effect of \lambda on the Learning Errors for a ' int2str(degree) ...
           '-Polynomial for ' title_str ' Cross Validation']);
    set(gca,'XScale','log') % Log scale
    set(gca,'YScale','log') % Log scale
    filename = [ 'img/lambda_sweep_' filename_k '_degree=' int2str(degree) '.pdf'];
    export_fig(filename);
end