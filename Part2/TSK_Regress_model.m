%% Fuzzy Systems - Regression Part 2
% Aforozi Thomais
% AEM: 9291
%% Clear Workspace
clear;
close all;
%% Read & Prepare dataset
fprintf('Load & prepare dataset... \n');
superconduct = csvread('train.csv',1,0);

[training_data,validation_data,check_data] = split_scale(superconduct,1);

%% Grid Search & 5-fold Cross Validation 
% number of features
NF = [4 6 10 12];
% radius of clusters (number of rules)
radii = [0.25 0.45 0.65 0.85];
   
meanError = zeros(length(NF),length(radii));

% feature selection
% relieff returns the most significant predictors
fprintf('Select features (relieff)... \n');
[ranks, weights] = relieff(training_data(:,1:end-1), training_data(:,end), 10);

observations = length(training_data(:,end));

fprintf('Loop... \n');
pointer = 1;
for f = 1 : length(NF)
    for r = 1 : length(radii)
        
        % 5-fold cross validation
        cv = cvpartition(observations, 'KFold', 5);
               
        for k = 1:5
            fprintf(['Repetition: ', num2str(pointer)]);
            fprintf('\n');
            % select features
            features_selected = [training_data(:,ranks(1:NF(f))) training_data(:,end)];
            
            % Getting the indices of the randomly splitted data with 5-fold cross
            % validation
            vali = test(cv,k); 
            train = training(cv,k);
            
            valIdx = find(vali == 1);
            trainIdx = find(train == 1);
            
            training_data_new = features_selected(trainIdx,:);
            validation_data_new = features_selected(valIdx,:);        
            
            % generate the fis model
            genfis_opt = genfisOptions('SubtractiveClustering',...
                    'ClusterInfluenceRange',radii(r));
            fis = genfis(training_data_new(:, 1:end-1),training_data_new(:,end),genfis_opt);

           % Tune the fis
           % Set the validation data to avoid overfitting
           AnfisOpt = anfisOptions('InitialFIS', fis, 'EpochNumber',10,'ValidationData', validation_data_new);
           AnfisOpt.DisplayANFISInformation = 0;
           AnfisOpt.DisplayErrorValues = 0;
           AnfisOpt.DisplayStepSize = 0;
           AnfisOpt.DisplayFinalResults = 0;
           
           [trainFis, trainError, ~ , chkFis, chkError] = anfis(training_data_new , AnfisOpt);
           
           meanError(f,r) =  meanError(f,r) + mean(chkError);
           pointer = pointer + 1;
        end
        
    end
end
 
meanError = meanError/5;

%% Mean Error plots
% Mean error respective to number of features
figure;
hold on;
grid on;
title('Mean error respective to number of features');
xlabel('Number of features')
ylabel('Mean Error')
plot(NF, meanError(:, 1),'LineWidth', 1.5);
plot(NF, meanError(:, 2),'LineWidth', 1.5);
plot(NF, meanError(:, 3),'LineWidth', 1.5);
plot(NF, meanError(:, 4),'LineWidth', 1.5);
legend('0.25 radius', '0.45 radius', '0.65 radius', '0.85 radius');

% Mean error respective to the cluster radius
figure;
hold on;
grid on;
title('Mean error respective to the cluster radius');
xlabel('Cluster radius')
ylabel('Mean Error')
plot(radii, meanError(1, :),'LineWidth', 1.5);
plot(radii, meanError(2, :),'LineWidth', 1.5);
plot(radii, meanError(3, :),'LineWidth', 1.5);
plot(radii, meanError(4, :),'LineWidth', 1.5);
legend('4 features', '6 features', '10 features','12 features');

%% Find the optimal NF & radius combination
fprintf('Find optimal values... \n');
[optimal_NF, optimal_radius] = find(meanError == min(meanError(:)));

des_features = NF(optimal_NF);
desired_rad = radii(optimal_radius);
fprintf(['Optimal NF: ', num2str(des_features)]);
fprintf(['\n Optimal radius: ', num2str(desired_rad)]);
fprintf('\n');

optimal_data = superconduct(:,ranks(1:des_features));
opt_training_data = [training_data(:,ranks(1:des_features)) training_data(:,end)];
opt_validation_data = [validation_data(:,ranks(1:des_features)) validation_data(:,end)];
opt_check_data = [check_data(:,ranks(1:des_features)) check_data(:,end)];

%% Generate & Train the final TSK model
fprintf('Generate the final TSK model... \n');
options_optAnfis = genfisOptions('SubtractiveClustering','ClusterInfluenceRange', desired_rad);
optimal_fis = genfis(opt_training_data(:,1:end-1),  opt_training_data(:,end),options_optAnfis);
fprintf('Train the final TSK model... \n');
anfisOpt = anfisOptions('InitialFIS', optimal_fis, 'EpochNumber',100,'ValidationData', opt_validation_data);
[trn_OptFis,trnOptError,~,valOptFis,valOptError] = anfis(opt_training_data,anfisOpt);

%% Evalutate the final TSK model
fprintf('Evaluation... \n');
y = evalfis(opt_check_data(:,1:end-1),valOptFis);
fprintf('Desired metrics: \n');
% Calculate the desired errors
% 1. MSE
n = length(y);
MSE = sum((opt_check_data(:,end) - y) .^ 2) / n;
RMSE = sqrt(MSE);

% 2. R^2
SSres = sum((opt_check_data(:,end) - y) .^ 2 );
SStot = sum((opt_check_data(:,end) - mean(opt_check_data(:,end))) .^ 2 );
R2 = 1 - SSres / SStot;

% 3. NMSE & NDEI
num = sum((opt_check_data(:,end) - y).^2);
den = sum((opt_check_data(:,end) - mean(opt_check_data(:,end))).^2);
NMSE = num/den;
NDEI = sqrt(NMSE);

fprintf('MSE: %f\n', MSE);
fprintf('RMSE: %f\n', RMSE);
fprintf('R^2: %f\n', R2);
fprintf('NMSE: %f\n', NMSE);
fprintf('NDEI: %f\n', NDEI);

%% Plots
% predictions & real values
figure;
hold on;
grid on;
title('Predictions and Real values','Interpreter','Latex');
xlabel('Test dataset sample','Interpreter','Latex');
ylabel('y','Interpreter','Latex');
plot(1:length(y), y,'o','Color','blue');
plot(1:length(y), opt_check_data(:, end),'+','Color','red');
leg = legend('Predictions', 'Real values');
set(leg,'Interpreter','latex');

% Learning curves
figure;
plot(trnOptError, 'LineWidth' , 1.2);
hold on;
plot(valOptError, 'LineWidth' , 1.2);
leg1 = legend('Training Error','Validation Error');
set(leg1,'Interpreter','latex');
xlabel('Number of Iterations','Interpreter','Latex'); 
ylabel('Error','Interpreter','Latex');

% membership functions of the input variables
% initial
for l = 1:length(optimal_fis.input) %size(opt_training_data,2)-1
    figure; 
    [xmf, ymf] = plotmf(optimal_fis, 'input', l);
    plot(xmf, ymf);
    xlabel('Input (initial)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(l)]);
end

% trained
for l = 1:length(trn_OptFis.input) %size(opt_training_data,2)-1
    figure; 
    [xmf, ymf] = plotmf(trn_OptFis, 'input', l);
    plot(xmf, ymf);
    xlabel('Input (trained)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(l)]);
end

% output
figure;
plot(y(1:100), 'LineWidth', 1.5); 
hold on;
plot(opt_check_data(1:100,end), 'LineWidth', 1.5);
leg = legend('TSK output','Real output');
set(leg,'Interpreter','latex');


% prediction error
figure;
hold on;
title('Prediction Error ');
xlabel('Check Dataset Sample');
ylabel('Squared Error');
plot(1:length(y), (y - opt_check_data(:,end)).^2 );