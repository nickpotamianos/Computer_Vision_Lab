%% partA_ex6.m
%
% Testing ECC & LK algorithms' robustness to additive noise
% Tests two types of noise:
% - Gaussian noise N(0,σ²) for σ² = 4, 8, 12 gray levels
% - Uniform noise U[-α,α] for α = 6^(1/3), 12^(1/3), 18^(1/3) gray levels

clear; clc; close all;

%% 1) Select two frames that are sufficiently apart
videoFile = 'video1_high.avi';  % or 'video2_high.avi'
if ~isfile(videoFile)
    error('Video file not found in current directory.');
end

vObj = VideoReader(videoFile);

frameIndexTemplate = 1;
frameIndexImage = 40;  % significantly later frame

if frameIndexImage > vObj.NumFrames
    error('Video has fewer than %d frames!', frameIndexImage);
end

frame1 = read(vObj, frameIndexTemplate);
frame2 = read(vObj, frameIndexImage);

% Convert to grayscale if needed
if size(frame1,3) == 3
    frame1 = rgb2gray(frame1);
end
if size(frame2,3) == 3
    frame2 = rgb2gray(frame2);
end

% Convert to double without additional normalization
template = double(frame1);
image = double(frame2);

%% 2) ECC/LK parameters
num_levels = 1;          
num_iterations = 15;       
transform = 'affine';  
init_warp = eye(2,3);   % Identity initialization

%% 3) Define noise parameters
gaussianVariances = [4, 8, 12];  % σ² values
uniformAlphas = nthroot([6, 12, 18], 3);  % α values

% Initialize results storage
numTrials = 100;  % Number of repetitions for each noise case
ResultsStruct = struct();
idxCase = 0;

%% 4) Run experiments with Gaussian noise
for varIdx = 1:length(gaussianVariances)
    sigma = sqrt(gaussianVariances(varIdx));
    
    % Arrays to store results for this noise level
    mse_ecc = zeros(1, numTrials);
    mse_lk = zeros(1, numTrials);
    rho_ecc = zeros(1, numTrials);
    
    disp(['=== Testing Gaussian noise with σ² = ', num2str(sigma^2), ' ===']);
    
    for trial = 1:numTrials
        % Add Gaussian noise to both images
        noisy_template = template + sigma * randn(size(template));
        noisy_image = image + sigma * randn(size(image));
        
        % Run ECC-LK alignment
        [res, res_lk, MSE, rho, MSELK] = ecc_lk(...
            noisy_image, ...
            noisy_template, ...
            num_levels, ...
            num_iterations, ...
            transform, ...
            init_warp);
        
        % Store results
        mse_ecc(trial) = MSE(end);
        mse_lk(trial) = MSELK(end);
        rho_ecc(trial) = rho(end);
        
        if mod(trial, 20) == 0
            disp(['Completed ', num2str(trial), ' trials']);
        end
    end
    
    % Store summary statistics
    idxCase = idxCase + 1;
    ResultsStruct(idxCase).type = 'Gaussian';
    ResultsStruct(idxCase).parameter = sigma^2;
    ResultsStruct(idxCase).MSE_ECC_mean = mean(mse_ecc);
    ResultsStruct(idxCase).MSE_ECC_std = std(mse_ecc);
    ResultsStruct(idxCase).MSE_LK_mean = mean(mse_lk);
    ResultsStruct(idxCase).MSE_LK_std = std(mse_lk);
    ResultsStruct(idxCase).rho_ECC_mean = mean(rho_ecc);
    ResultsStruct(idxCase).rho_ECC_std = std(rho_ecc);
end

%% 5) Run experiments with Uniform noise
for alphaIdx = 1:length(uniformAlphas)
    alpha = uniformAlphas(alphaIdx);
    
    % Arrays to store results for this noise level
    mse_ecc = zeros(1, numTrials);
    mse_lk = zeros(1, numTrials);
    rho_ecc = zeros(1, numTrials);
    
    disp(['=== Testing Uniform noise with α = ', num2str(alpha), ' ===']);
    
    for trial = 1:numTrials
        % Add uniform noise to both images
        noisy_template = template + (2*alpha) * (rand(size(template)) - 0.5);
        noisy_image = image + (2*alpha) * (rand(size(image)) - 0.5);
        
        % Run ECC-LK alignment
        [res, res_lk, MSE, rho, MSELK] = ecc_lk(...
            noisy_image, ...
            noisy_template, ...
            num_levels, ...
            num_iterations, ...
            transform, ...
            init_warp);
        
        % Store results
        mse_ecc(trial) = MSE(end);
        mse_lk(trial) = MSELK(end);
        rho_ecc(trial) = rho(end);
        
        if mod(trial, 20) == 0
            disp(['Completed ', num2str(trial), ' trials']);
        end
    end
    
    % Store summary statistics
    idxCase = idxCase + 1;
    ResultsStruct(idxCase).type = 'Uniform';
    ResultsStruct(idxCase).parameter = alpha;
    ResultsStruct(idxCase).MSE_ECC_mean = mean(mse_ecc);
    ResultsStruct(idxCase).MSE_ECC_std = std(mse_ecc);
    ResultsStruct(idxCase).MSE_LK_mean = mean(mse_lk);
    ResultsStruct(idxCase).MSE_LK_std = std(mse_lk);
    ResultsStruct(idxCase).rho_ECC_mean = mean(rho_ecc);
    ResultsStruct(idxCase).rho_ECC_std = std(rho_ecc);
end

%% 6) Display Results
disp('====== Summary of Results ======');
for i = 1:length(ResultsStruct)
    disp(['Case #', num2str(i)]);
    disp(['Noise Type: ', ResultsStruct(i).type]);
    if strcmp(ResultsStruct(i).type, 'Gaussian')
        disp(['σ² = ', num2str(ResultsStruct(i).parameter)]);
    else
        disp(['α = ', num2str(ResultsStruct(i).parameter)]);
    end
    disp(['ECC MSE: ', num2str(ResultsStruct(i).MSE_ECC_mean), ' ± ', num2str(ResultsStruct(i).MSE_ECC_std)]);
    disp(['LK MSE: ', num2str(ResultsStruct(i).MSE_LK_mean), ' ± ', num2str(ResultsStruct(i).MSE_LK_std)]);
    disp(['ECC ρ: ', num2str(ResultsStruct(i).rho_ECC_mean), ' ± ', num2str(ResultsStruct(i).rho_ECC_std)]);
    disp('------------------------');
end

%% 7) Create visualization plots
figure('Name', 'Noise Analysis Results');

% Plot MSE comparison
subplot(2,1,1);
x = 1:length(ResultsStruct);
errorbar(x-0.1, [ResultsStruct.MSE_ECC_mean], [ResultsStruct.MSE_ECC_std], 'b-o', 'DisplayName', 'ECC');
hold on;
errorbar(x+0.1, [ResultsStruct.MSE_LK_mean], [ResultsStruct.MSE_LK_std], 'r-o', 'DisplayName', 'LK');
hold off;
ylabel('MSE');
title('Algorithm Performance under Different Noise Conditions');
legend('Location', 'best');
grid on;

% Add custom x-tick labels
xticks(1:length(ResultsStruct));
xticklabels(arrayfun(@(x) sprintf('%s\n%s=%.2f', ...
    x.type, iif(strcmp(x.type,'Gaussian'),'σ²','α'), x.parameter), ...
    ResultsStruct, 'UniformOutput', false));
xtickangle(45);

% Plot rho values for ECC
subplot(2,1,2);
errorbar(x, [ResultsStruct.rho_ECC_mean], [ResultsStruct.rho_ECC_std], 'g-o');
ylabel('\rho');
title('ECC Correlation Coefficient under Different Noise Conditions');
xticks(1:length(ResultsStruct));
xticklabels(arrayfun(@(x) sprintf('%s\n%s=%.2f', ...
    x.type, iif(strcmp(x.type,'Gaussian'),'σ²','α'), x.parameter), ...
    ResultsStruct, 'UniformOutput', false));
xtickangle(45);
grid on;

% Helper function for ternary operator
function out = iif(condition, trueVal, falseVal)
    if condition
        out = trueVal;
    else
        out = falseVal;
    end
end