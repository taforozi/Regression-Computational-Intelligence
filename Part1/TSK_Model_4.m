%% Fuzzy Systems - Regression Part 1
% Aforozi Thomais
% 9291
% 3 Membership functions - Polynomial

function [fis,MSE,RMSE,R2,NMSE,NDEI,y] = TSK_Model_4(training_data,validation_data,check_data)
%% Generate Tsk Model
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = 3;
opt.InputMembershipFunctionType = "gbellmf";
opt.OutputMembershipFunctionType = "linear";

fis = genfis(training_data(:,1:end-1),training_data(:,end),opt);

%% Train TSK Model
options = anfisOptions('InitialFIS', fis, 'EpochNumber', 100,...
    'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'ValidationData', validation_data,...
    'OptimizationMethod', 1);

[trnFis, trnError, ~, valFis, valError] = anfis(training_data, options);

%% Evaluate trained model
y = evalfis(check_data(:,1:end-1), valFis);

%% Calculate errors
% 1. MSE
N = length(y);
MSE = sum((check_data(:,end) - y) .^ 2) / N;
% MSE = mse(y,chkData(:,end));
RMSE = sqrt(MSE);

% 2. R^2
SSres = sum((check_data(:,end) - y) .^ 2 );
SStot = sum((check_data(:,end) - mean(check_data(:,end))) .^ 2 );
R2 = 1 - SSres / SStot;

% 3. NMSE & NDEI
num = sum((check_data(:,end) - y).^2);
den = sum((check_data(:,end) - mean(check_data(:,end))).^2);
NMSE = num/den;
NDEI = sqrt(NMSE);

fprintf('Metrics for TSK Model 4 \n');
fprintf('MSE: %f\n', MSE);
fprintf('RMSE: %f\n', RMSE);
fprintf('R^2: %f\n', R2);
fprintf('NMSE: %f\n', NMSE);
fprintf('NDEI: %f\n', NDEI);

%% Plots
% initial Membership functions of input variables
for l = 1:length(fis.input)
    figure; 
    [xmf, ymf] = plotmf(fis, 'input', l);
    plot(xmf, ymf,'LineWidth',1.5);
    xlabel('Model 4 - Input (initial)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(l)]);
end
% Trained Membership functions of input variables
for l = 1:length(trnFis.input)
    figure; 
    [xmf, ymf] = plotmf(trnFis, 'input', l);
    plot(xmf, ymf,'LineWidth',1.5);
    xlabel('Model 4 - Input (trained)', 'Interpreter', 'Latex');
    ylabel('Degree of membership', 'Interpreter', 'Latex');
    title(['Input #' num2str(l)]);
end

% Learning curves
figure;
plot([trnError valError], 'LineWidth', 1.5);
title('Learning curves for TSK model 4','Interpreter','Latex'); 
grid on;
xlabel('Number of Iterations','Interpreter','Latex'); 
ylabel('Error','Interpreter','Latex');
leg1 = legend('Training Error','Validation Error');
set(leg1,'Interpreter','latex');

% Prediction error
figure;
subplot(1,2,1);
plot(y(1:100), 'LineWidth', 1.5); hold on;
plot(check_data(1:100,end), 'LineWidth', 1.5);
leg = legend('$$\hat{y}$$','y');
set(leg,'Interpreter','latex');
title('Model 4 - Output','Interpreter','latex');
xlabel('Number of Iterations','Interpreter','Latex'); 
ylabel('y, $$\hat{y}$$','Interpreter','Latex');
subplot(1,2,2);
plot(abs(check_data(1:100,end) - y(1:100)), 'LineWidth', 1.5);
title('Model 4 - Prediction error','Interpreter','latex');
xlabel('Number of Iterations','Interpreter','Latex'); 
ylabel('Error','Interpreter','Latex');

% predictions & real values
figure;
hold on;
title('Model 4 - Predictions and Real values','Interpreter','Latex');
xlabel('Test dataset sample','Interpreter','Latex');
ylabel('y','Interpreter','Latex');
plot(1:length(y), y,'o','Color','blue');
plot(1:length(y), check_data(:, end),'+','Color','red');
leg = legend('Predictions', 'Real values');
set(leg,'Interpreter','latex');
end