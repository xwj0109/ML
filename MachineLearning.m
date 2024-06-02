
% ****************************
%  IF2211 Coursework
%  author: Chuanping Sun
%  Bayes Business School
%  City University of London
% ****************************

clear all; clc; warning off;


%% readtable / treat missing values / specify import types
opts = detectImportOptions('trainData.csv');
opts.MissingRule = 'omitrow'; % missing values can cause problematic estimation results
                              % some functions are robust with missing data,
                              % while some are not

categoricalVariableNames = {'grade', 'emp_length', 'home_ownership',  ...
       'collections_12_mths_ex_med', 'application_type', 'acc_now_delinq'};
opts = setvartype(opts, categoricalVariableNames, 'categorical' );
                    
opts = setvaropts(opts, 'emp_length', 'TreatAsMissing', {'n/a'});

% remove variables that have too many missing values
miss_var = {'mths_since_last_delinq'}; % this variable has 110000+ missing data
selected = setdiff(opts.VariableNames(3:end), miss_var);
opts.SelectedVariableNames = selected;

% readtable with options
trainTbl = readtable('trainData.csv', opts);

%convert loan_status to logical values
key = 'Charged Off';
trainTbl.loan_status = strcmpi(trainTbl.loan_status, key);

summary(trainTbl)

% check variable levels to determine which variables are categorical 
func = @(x) numel(unique(x));
varLevel = varfun(func,trainTbl); 

% -------------------------------------------------------------------------
% similarly for testdata

testTbl = readtable('testData.csv', opts);

%convert loan_status to logical values
testTbl.loan_status = strcmpi(testTbl.loan_status, key);

summary(testTbl)
% -------------------------------------------------------------------------
% group predictor variables
predictorNames = setdiff(trainTbl.Properties.VariableNames, 'loan_status');
numVariableNames = setdiff(predictorNames, categoricalVariableNames); % numerical predictors

%% convert categorical variables to dummy variables 
% because some functions such as corr(), ridge() and lassoglm() cannot deal with categorical predictors

% create dummy variable for 'grade'
grade_dum = dummyvar(trainTbl.grade); 
grade_dum2 = dummyvar(testTbl.grade);
types = categories(trainTbl.grade);
dum_names = join([repmat({'grade'}, length(types), 1),types], '_' ); 
grade_trainTbl = array2table(grade_dum, 'VariableNames', dum_names);
grade_testTbl = array2table(grade_dum2, 'VariableNames', dum_names);

% create dummy variables for 'emp_length'
emp_length_dum = dummyvar(trainTbl.emp_length);
emp_length_dum2 = dummyvar(testTbl.emp_length);
types = categories(trainTbl.emp_length);
dum_names = join ([repmat({'emp_length'}, length(types), 1),types], '_' );  
emp_length_trainTbl = array2table(emp_length_dum, 'VariableNames', dum_names);
emp_length_testTbl = array2table(emp_length_dum2, 'VariableNames', dum_names);

% create dummy variable for 'home_ownership'
home_ownership_dum = dummyvar(trainTbl.home_ownership);
home_ownership_dum2 = dummyvar(testTbl.home_ownership);
types = categories(trainTbl.home_ownership);
dum_names = join([repmat({'home_ownership'}, length(types), 1),types], '_' ); 
home_ownership_trainTbl = array2table(home_ownership_dum, 'VariableNames', dum_names);
home_ownership_testTbl = array2table(home_ownership_dum2, 'VariableNames', dum_names);

% create dummy variable for 'collections_12_mths_ex_med'
collections_12_mths_ex_med_dum = dummyvar(trainTbl.collections_12_mths_ex_med);
collections_12_mths_ex_med_dum2 = dummyvar(testTbl.collections_12_mths_ex_med);
types = categories(trainTbl.collections_12_mths_ex_med);
dum_names = join([repmat({'collections_12_mths_ex_med'}, length(types), 1),types], '_' ); 
collections_12_mths_ex_med_trainTbl = array2table(collections_12_mths_ex_med_dum, 'VariableNames', dum_names);
collections_12_mths_ex_med_testTbl = array2table(collections_12_mths_ex_med_dum2, 'VariableNames', dum_names);

% create dummy variable for 'application_type'
application_type_dum = dummyvar(trainTbl.application_type);
application_type_dum2 = dummyvar(testTbl.application_type);
types = categories(trainTbl.application_type);
dum_names = join([repmat({'application_type'}, length(types), 1),types], '_' ); 
application_type_trainTbl = array2table(application_type_dum, 'VariableNames', dum_names);
application_type_testTbl = array2table(application_type_dum2, 'VariableNames', dum_names);

% create dummy variable for 'acc_now_delinq'
acc_now_delinq_dum = dummyvar(trainTbl.acc_now_delinq);
acc_now_delinq_dum2 = dummyvar(testTbl.acc_now_delinq);
types = categories(trainTbl.acc_now_delinq);
dum_names = join([repmat({'acc_now_delinq'}, length(types), 1),types], '_' ); 
acc_now_delinq_trainTbl = array2table(acc_now_delinq_dum, 'VariableNames', dum_names);
acc_now_delinq_testTbl = array2table(acc_now_delinq_dum2, 'VariableNames', dum_names);


% create predictor table. Note that first column of dummy variables are
% used as reference level

Predictor_trainTbl = [trainTbl(:,numVariableNames), grade_trainTbl(:,2:end), ...
               emp_length_trainTbl(:,2:end), home_ownership_trainTbl(:,2:end), ...
               collections_12_mths_ex_med_trainTbl(:,2:end), ...
               application_type_trainTbl(:,2:end), acc_now_delinq_trainTbl(:,2:end) ];
           
Predictor_testTbl = [testTbl(:,numVariableNames), grade_testTbl(:,2:end), ...
               emp_length_testTbl(:,2:end), home_ownership_testTbl(:,2:end), ...
               collections_12_mths_ex_med_testTbl(:,2:end), ...
               application_type_testTbl(:,2:end), acc_now_delinq_testTbl(:,2:end) ];

new_trainTbl = [Predictor_trainTbl, trainTbl(:,{'loan_status'})];
new_testTbl = [Predictor_testTbl, testTbl(:,{'loan_status'})];


%% preliminary analysis / covariance analysis
% In this block, using the new_trainTbl to analyse the correlation between
% predicors and the response variable 'loan_status'. Use a heatmap to
% display the correlation structure. And find out the top 10 and bottom 10
% correlated predictors, call them top10 and bottom10. Include the graph in
% your reporting. 

c = corr(new_trainTbl{:, :})
vnames = new_trainTbl.Properties.VariableNames(:);

% heatmap
figure 
imagesc(c)   % display the matrix using color scales
colorbar;
set(gca,'XTick',1:length(vnames), 'XTickLabel', vnames,'XTickLabelRotation',45,  ...
         'YTick', 1:length(vnames), 'YTickLabel', vnames, 'TickLabelInterpreter', 'none');

title('Correlation-coefficient colormap');
saveas(gca, 'correlation_coefficient_colormap.png')

% finding top/bottom 10s, note: absolute values 

c_vector = c(:,end);
[sorted_coefficients, sorted_indices] = sort(abs(c_vector), 'descend');
top10 = vnames(sorted_indices(2:11)); % the first one is y itself
bottom10 = vnames(sorted_indices(end-9:end)); % the last is the most irrelevant predictor

%% GLS 
% estimate a logistic model using the trainTbl (i.e., contains categorical variables)
% hint: use the function fitglm().
% #########################################################################
% -------------------------------------------------------------------------


% moving loan_status to the last column for modeling in both test/train
% tables
columnIndex = find(strcmp(trainTbl.Properties.VariableNames, 'loan_status'));
trainTbl= [trainTbl(:,1:columnIndex-1), trainTbl(:,columnIndex+1:end), trainTbl(:,columnIndex)];
glsmdl = fitglm(trainTbl, "Distribution", "binomial", "Link", "logit");

columnIndex = find(strcmp(testTbl.Properties.VariableNames, 'loan_status'));
testTbl= [testTbl(:,1:columnIndex-1), testTbl(:,columnIndex+1:end), testTbl(:,columnIndex)];

% -------------------------------------------------------------------------
% #########################################################################

% forecast using training data, and compute glsMSE_train
% #########################################################################
% -------------------------------------------------------------------------

glspredict1 = predict(glsmdl, trainTbl(:, 1:end-1));
glspredict1 = (glspredict1 > 0.5);
glsMSE_train = mean((glspredict1 - trainTbl.loan_status).^2);

% -------------------------------------------------------------------------
% #########################################################################

% forecast using test data, and compute glsMSE_test
% #########################################################################
% -------------------------------------------------------------------------

glspredict2 = predict(glsmdl, testTbl(:, 1:end-1));
glspredict2 = (glspredict2 > 0.5);
glsMSE_test = mean((glspredict2 - testTbl.loan_status).^2);

% -------------------------------------------------------------------------
% #########################################################################

% let's put the result into a confusion matrix 
%[mdl0_confM, mdl0_acc] = confusionmat(new_testTbl.loan_status, glspredict2);
%confusionchart(mdl0_confM, mdl0_acc);

% accuracy of the gls model
%numCorrectPred = sum(diag(mdl0_confM));
%totalPred = sum(mdl0_confM(:));
%ACC0 = (numCorrectPred/totalPred);

%% stepwise regression models
% Fit a stepwise regression model using only the top10 predictors you have
% found in the covariance analysis above. 
swMdl = stepwiseglm(new_trainTbl(:,[top10; {'loan_status'}]),  'constant', 'upper', 'linear', ...
              'Distribution', 'binomial' );

% forecast using training sample, and compute MSE for the training sample
swForecast_train = predict(swMdl, Predictor_trainTbl(:,top10));
swForecast_train = (swForecast_train >=0.5); 
swMSE_train = mean((swForecast_train - new_trainTbl.loan_status).^2);
fprintf('MSE of the step-wise estimator using training sample is: %6.4f \n', swMSE_train);


% forecast using testing sample, and compute MSE for the testing sample, 
% call it swMSE_test.
% #########################################################################
% -------------------------------------------------------------------------

swForecasting_test = predict(swMdl, Predictor_testTbl(:, top10));
swForecasting_test = (swForecasting_test >= 0.5);
swMSE_test = mean((swForecasting_test - testTbl.loan_status).^2);


% -------------------------------------------------------------------------
% #########################################################################

% also let's fit in a ConfusionMat for the testing sample
%[mdl1_confM, mdl1_acc] = confusionmat(testTbl.loan_status, swForecasting_test);
%confusionchart(mdl1_confM, mdl1_acc);

%numCorrectPred = sum(diag(mdl1_confM));
%totalPred = sum(mdl1_confM(:));
%ACC1 = (numCorrectPred/totalPred);

%% lasso
% model estimation using training data
% lasso cannot handle categorical predictors, thus, we use the
% dummy-variable tbl
[B_lasso,fitInfo] = lassoglm(Predictor_trainTbl{:,:}, trainTbl.loan_status, 'binomial', 'CV', 3);

B0 = fitInfo.Intercept(fitInfo.IndexMinDeviance);
lasso_bhat = [B0; B_lasso(:, fitInfo.IndexMinDeviance)];

% predict loan_status using training sample
lassohat_train = glmval(lasso_bhat, Predictor_trainTbl{:,:}, 'logit'); % for the binomial response
lassoForecast_train = (lassohat_train >=0.5);
lassoMSE_train = mean((lassoForecast_train -trainTbl.loan_status).^2);

% output confusion chart for training sample

figure()

c_lasso_train = confusionchart(trainTbl.loan_status, lassoForecast_train);
title('LASSO confusion chart using training data')

% checking the ACC for training data
[mdl2_confM, mdl2_acc] = confusionmat(trainTbl.loan_status, lassoForecast_train);
numCorrectPred = sum(diag(mdl2_confM));
totalPred = sum(mdl2_confM(:));
lassoACC_train= (numCorrectPred/totalPred);
 

% predict loan_status using testing sample. Hint: follow the method we used
% for training sample
% #########################################################################
% -------------------------------------------------------------------------

lassohat_test = glmval(lasso_bhat, Predictor_testTbl{:, :}, 'logit');
lassoForecast_test = (lassohat_test >= 0.5);
lassoMSE_test = mean((lassoForecast_test - testTbl.loan_status).^2);
fprintf('MSE of the lasso estimator using testing sample is: %6.4f \n', lassoMSE_test);
% -------------------------------------------------------------------------
% #########################################################################
% output confusion chart for testing sample
figure()
c_lasso_test = confusionchart(testTbl.loan_status, lassoForecast_test);
title('LASSO confusion chart using testing data')

% checking the ACC for testing data
[mdl2_confM, mdl2_acc] = confusionmat(testTbl.loan_status, lassoForecast_test);
numCorrectPred = sum(diag(mdl2_confM));
totalPred = sum(mdl2_confM(:));
lassoACC_test = (numCorrectPred/totalPred)

% plot the lasso coefficient (B_lasso)along lambda values, and save the figure 
% as png file for reporting. You don't need to standarise the dataset.

% #########################################################################
% -------------------------------------------------------------------------
Regnames = ['Intercept', testTbl.Properties.VariableNames];
figure()
plot(fitInfo.Lambda, B_lasso, "LineWidth", 2);
legend(Regnames, 'Interpreter', 'none');
xlabel("Lambda Value");
ylabel("Estimated Coefficients");
title('Unscaled LASSO with intercept')
saveas(gca, 'Unscaled LASSO with intercept.png')

% -------------------------------------------------------------------------
% #########################################################################



%% bagging 
% fit an ensemble model using bagging algo (using a tree as the weak
% learner. output the estimated model as 'bagMdl'.) Set 5 maximum splits
% for each tree. Use interaction-curvature for predictor selection
% criterion. 

% #########################################################################
% -------------------------------------------------------------------------

t = templateTree('MaxNumSplits', 20, 'PredictorSelection', 'interaction-curvature', 'Reproducible', true);
bagMdl = fitcensemble(trainTbl, 'loan_status', 'Learners', t, ...
                      'CategoricalPredictors', categoricalVariableNames, 'Method', 'Bag');

% -------------------------------------------------------------------------
% #########################################################################
                 
% forecast using training sample and find MSE for the training sample (save it as bagMSE_train).
% #########################################################################
% -------------------------------------------------------------------------

bagForecast_train = predict(bagMdl, trainTbl(:, 1:end-1));
bagMSE_train = mean((bagForecast_train - trainTbl.loan_status).^2);

% -------------------------------------------------------------------------
% #########################################################################
fprintf('MSE of the bagging ensemble estimator using training sample is: %6.4f \n', bagMSE_train); 
% forecast using testing sample, and find MSE for the test sample (save it
% as bagMSE_test).
% #########################################################################
% -------------------------------------------------------------------------

bagForecast_test = predict(bagMdl, testTbl(:,1:end-1));
bagMSE_test = mean((bagForecast_test - testTbl.loan_status).^2);

% -------------------------------------------------------------------------
% #########################################################################
fprintf('MSE of the bagging ensemble estimator using testing sample is: %6.4f \n', bagMSE_test); 

% predictor importance estimation 
% note 'PredictorSelection','interaction-curvature' ensures importance is
% not baised towards variables with many levels
% plot a bar chat to show predictors' importance
% #########################################################################
% -------------------------------------------------------------------------


imp = bagMdl.predictorImportance;  
figure;
bar(imp);
title('Predictor Importance Bagging');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTick = 1:length(bagMdl.PredictorNames);
h.XTickLabel = bagMdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% -------------------------------------------------------------------------
% #########################################################################

% find top 10 important predictors
bagImportanceTbl = array2table(abs(imp)', 'RowNames', bagMdl.PredictorNames, 'VariableNames', {'abs_coef'});
bagImportanceTbl = sortrows(bagImportanceTbl);
bagTop10 = bagImportanceTbl.Properties.RowNames(end-9:end);
disp(bagTop10)




%% boosting
% fit a boosting ensemble model using a tree as the weak learner.
% It is very similar to the procedure above. Save MSE for the training and
% testing sample as boostMSE_train and boostMSE_test, respectively. Find
% the top 10 important predictors.
% #########################################################################
% -------------------------------------------------------------------------

% fit in the adaboostM1 model & compute MSE for training/testing sample
boostMdl = fitcensemble(trainTbl, 'loan_status', 'Learners', t, ...
    'Categorical', categoricalVariableNames, 'Method','AdaBoostM1');

boostForecast_train = predict(boostMdl, trainTbl(:, 1:end-1));
boostMSE_train = mean((boostForecast_train - trainTbl.loan_status).^2);
fprintf('MSE of the adaptive boosting estimator using training sample is: %6.4f \n', boostMSE_train); 
boostForecast_test = predict(boostMdl, testTbl(:, 1:end-1));
boostMSE_test = mean((boostForecast_test - testTbl.loan_status).^2);
fprintf('MSE of the adaptive boosting estimator using testing sample is: %6.4f \n', boostMSE_test); 

% top 10 predictors & visualisation
imp2 = boostMdl.predictorImportance;
figure;
bar(imp);
title('Predictor Importance Boosting');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTick = 1:length(bagMdl.PredictorNames);
h.XTickLabel = bagMdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

boostImportanceTbl = array2table(abs(imp2)', 'RowNames', boostMdl.PredictorNames, ...
                                'VariableNames',{'abs_coef'});
boostImportanceTbl = sortrows(boostImportanceTbl);
boostTop10 = boostImportanceTbl.Properties.RowNames(end-9:end);
disp(boostTop10)

% -------------------------------------------------------------------------
% #########################################################################



%% random forest   
% fit a random forest with 50 trees. Randomly select 1/3 of total
% predictors to build each tree. Name this model rfMdl.
% #########################################################################
% -------------------------------------------------------------------------

rfMdl = TreeBagger(50, trainTbl, 'loan_status', 'NumPredictorsToSample',...
                    floor(size(trainTbl, 2)/3), 'OOBPrediction','on', ...
                    'OOBPredictorImportance', 'on', ...
                    'Method', 'classification', 'CategoricalPredictors', categoricalVariableNames, ...
                    'PredictorSelection', 'interaction-curvature', 'Reproducible', true);

% -------------------------------------------------------------------------
% #########################################################################
% MSE using training sample
rfForecast_train = predict(rfMdl, trainTbl(:,predictorNames)); % output is a cell array -> convert to numeric vector
rfForecast_train = cellfun(@str2double, rfForecast_train);  
rfMSE_train = mean((rfForecast_train - trainTbl.loan_status).^2);
% MSE using testing sample
rfForecast_test = predict(rfMdl, testTbl(:,predictorNames)); 
rfForecast_test = cellfun(@str2double, rfForecast_test);  
rfMSE_test = mean((rfForecast_test - testTbl.loan_status).^2);

% Predictor importance estimation
% #########################################################################
% -------------------------------------------------------------------------

imp3 = rfMdl.OOBPermutedPredictorDeltaError;
figure;
bar(imp3);
title('Predictor Importance Random Forest');
ylabel('Predictor importance estimates');
xlabel('Predictors');
h = gca;
h.XTick = 1:length(rfMdl.PredictorNames);
h.XTickLabel = rfMdl.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% -------------------------------------------------------------------------
% #########################################################################

%% save data
save('assignment.mat', 'trainTbl','testTbl', 'top10', 'bottom10','glsMSE_train', 'glsMSE_test', ...
    'swMSE_train', 'swMSE_test', ...
    'lassoMSE_train', 'lassoMSE_test', 'bagMSE_train', 'bagMSE_test', 'boostMSE_train', 'boostMSE_test',...
    'rfMSE_train', 'rfMSE_test'); % change the .mat file name as your student ID 








