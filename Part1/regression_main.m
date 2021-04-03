%% Fuzzy Systems - Regression Part 1
% Aforozi Thomais - AEM 9291
% Train 4 TSK models
% 2&3 Membership functions - Singleton
% 2&3 Membership functions - Polynomial
%% Clear workspace
clear;
close all;
%% Load & Prepare the Data
load airfoil_self_noise.dat
[training_data,validation_data,check_data] = split_scale(airfoil_self_noise,1);

%% Model 1 - 2 Membership functions (Singleton)
fprintf('Generate & train model 1... \n');
[fis_1,MSE_1,RMSE_1,R2_1,NMSE_1,NDEI_1,y_1] = TSK_Model_1(training_data,validation_data,check_data);

%% Model 2 - 3 Membership functions (Singleton)
fprintf('Generate & train model 2... \n');
[fis_2,MSE_2,RMSE_2,R2_2,NMSE_2,NDEI_2,y_2] = TSK_Model_2(training_data,validation_data,check_data);

%% Model 3 - 2 Membership functions (Polynomial)
fprintf('Generate & train model 3... \n');
[fis_3,MSE_3,RMSE_3,R2_3,NMSE_3,NDEI_3,y_3] = TSK_Model_3(training_data,validation_data,check_data);

%% Model 4 - 3 Membership functions (Polynomial)
fprintf('Generate & train model 4... \n');
[fis_4,MSE_4,RMSE_4,R2_4,NMSE_4,NDEI_4,y_4] = TSK_Model_4(training_data,validation_data,check_data);

