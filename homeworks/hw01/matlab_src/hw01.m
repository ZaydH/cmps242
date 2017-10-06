function hw01(degree, lambdas, k, ignoreBias, offset_term)
    % Performs the curve fitting in homework 1.
    %
    % Course Information:
    %    * University of California Santa Cruz
    %    * Prof. Manfred K. Warmuth
    %    * Fall 2017
    %    * https://classes.soe.ucsc.edu/cmps242/Fall17/
    %
    % Required Arguments:
    %    degree - Order of the polynomial
    %    lambdas - List of lambda values to test
    %    k - Number of cross validation folds
    %    ignoreBias - Boolean to determine whether to penalizes the w0
    %                 term in regularization.
    %
    % Optional Arguments:
    %    offset_term - Default "0".  Tests shifting the data.

    REQUIRED_ARGUMENT_COUNT = 4;

    % Disable singular warning but be a good citizen and restore original
    % state at the end.
    orig_state = warning;
    warning('off','MATLAB:nearlySingularMatrix');

    train_data=importdata("train.txt");
    test_data=importdata("test.txt");

    if nargin > REQUIRED_ARGUMENT_COUNT && ~isempty(offset_term)
        train_data(:,2) = train_data(:,2) + offset_term;
        test_data(:,2) = test_data(:,2) + offset_term;
    end

    for test_lambda=lambdas
        compare_train_test_single_lambda(train_data, test_data, degree, test_lambda, ignoreBias)
    end

    % K fold
    sweep_lambda(train_data, test_data, degree, lambdas, k, ignoreBias)
    % Leave One Out
    sweep_lambda(train_data, test_data, degree, lambdas, size(train_data,1), ignoreBias)

    % Restore original warning state
    warning(orig_state);
end


function compare_train_test_single_lambda(train_data, test_data, degree, lambda, ignoreBias)

   [~,~,~,y_train,y_test]=perform_cross_validation(train_data(:,1),train_data(:,2), ...
                                                   test_data(:,1),test_data(:,2), ...
                                                   degree, lambda, 0, ignoreBias);
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
    
    % Format the plot
    xlabel('X');
    ylabel('Target');
    title(['Learner Output for ' int2str(degree) ...
           '-Polynomial with \lambda=' generate_lambda_string(lambda)]);
    set(gcf, 'Color', 'w'); % Make the background white
    
    % Construct the graph without compressing the axes
    file_folder = 'img/';
    if ~exist(file_folder,'dir') mkdir(file_folder); end %#ok<SEPEX>
    file_name = ['test_train_compare_d=' int2str(degree) ...
                 '_lambda=' generate_lambda_string(lambda)];
    file_extension = '.pdf';
    full_file_path = [ file_folder file_name file_extension ];
    set(gca,'FontSize', 26);
    set(leg,'Location','Best')  % Prevent legend overlap with the data
    export_fig(full_file_path)
    
    % Allow for smart y-axis so the extremes do not dominate.
    if degree > 10
        ymin_fitted = min(train_data(:,2), train_data(:,2)) - 0.5; 
        ymax_fitted = max(train_data(:,2), train_data(:,2)) + 0.5; 
        if min(y_test) < ymin_fitted || max(y_test) > ymax_fitted
            filename_comp = [file_name '_fitted'];
            full_file_path = [ file_folder filename_comp file_extension ];
            ylim([ymin_fitted ymax_fitted])
            set(leg,'Location','Best')  % Legend location needs to be reset now the axes changed.
            export_fig(full_file_path)
        end
    end
    
%     % Create a histogram of the squared error.
%     numOfBins = 100;
%     [histFreq, histXout] = hist((y_test-test_data(:,2)).^2, numOfBins);
%     bar(histXout, histFreq/sum(histFreq)*100);
%     set(gca,'FontSize', 26);
%     ylabel('Frequency (%)');
%     file_name = ['test_err_hist=' int2str(degree) ...
%                  '_lambda=' generate_lambda_string(lambda)];
%     xlabel('Test Sample Squared Error')
%     full_file_path = [ file_folder file_name file_extension ];
%     export_fig(full_file_path)
end


% Used to make cleaner printing of the lambda value in the
% title of graphs.
function lambda_str=generate_lambda_string(lambda)
    if lambda == 0
        lambda_str = num2str(lambda);
        return;
    end
    power = round(log2(lambda));
    if power ~= 1
        lambda_str = [ '2^{' num2str(power) '}'];
    else
        lambda_str = '2'; % Prevent unnecessary power when lambda = 2
    end 
end 


function sweep_lambda(train_data, test_data, degree, lambdas, k, ignoreBias)
    % Get the cross validation errors
    [train_err,valid_err,test_err,~,~]=perform_cross_validation(train_data(:,1),train_data(:,2), ...
                                                               test_data(:,1),test_data(:,2), ...
                                                               degree, lambdas, k, ignoreBias);

    % Get the average errors
    train_avg_err = [mean(train_err,2), var(train_err')']; %#ok<UDIM>
    valid_avg_err = [mean(valid_err,2), var(valid_err')']; %#ok<UDIM>

    % Plot the results
    if (k~=size(train_data,1)); folds=k; else; folds=NaN; end
    plot_lambda_sweep(degree, lambdas, train_avg_err, valid_avg_err, test_err, folds);
end


function plot_lambda_sweep(degree, lambdas, train_avg_errs, valid_avg_errs, test_errs, k)
    
    y_neg = build_neg_error_bar(train_avg_errs,.01);
    errorbar(lambdas,train_avg_errs(:,1),y_neg,train_avg_errs(:,2));
    hold on; % Allow multiple plots simultaneously
    y_neg = build_neg_error_bar(valid_avg_errs,.01);
    errorbar(lambdas,valid_avg_errs(:,1),y_neg,valid_avg_errs(:,2));
    plot(lambdas, test_errs);
    hold off;
    leg = legend('Train','Validation','Test');
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
    title({['Effect of \lambda on Learning a ' int2str(degree) '-Degree'], ...
           ['Polynomial with ' title_str ' Cross Validation']});
    set(gca,'XScale','log') % Log scale
    set(gca,'YScale','log') % Log scale
    
    % Create the image
    image_path = 'img/';
    if ~exist(image_path,'dir') mkdir(image_path); end %#ok<SEPEX>
    filename = ['lambda_sweep_' filename_k '_degree=' int2str(degree)];
    file_ext = '.pdf';
    full_file_path = [ image_path filename file_ext];
    set(gca,'FontSize', 18);
    set(leg,'Location','Best')  % Prevent legend overlap with the data
    export_fig(full_file_path);
    
    % Export the data
    all_errs = [lambdas' train_avg_errs(:,1) valid_avg_errs(:,1)  test_errs];
    file_ext = '.csv';
    full_file_path = [ filename file_ext];
    dlmwrite(full_file_path, all_errs', ',');
end


% Matlab is clipping the negative bars
% So force it to appear "near" zero.
function min_err=build_neg_error_bar(err_vals,min_val)
    min_err=min(err_vals(:,1)-min_val, err_vals(:,2));
end 
