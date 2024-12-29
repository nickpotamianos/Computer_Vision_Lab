% Script for studying ECC and LK image alignment algorithms
close all;  % Close any existing figures

% 1. Load video and prepare initial frames
video = VideoReader('video1_low.avi');
template = readFrame(video);
image = readFrame(video);

% Convert to double and normalize
template = double(template);
image = double(image);
template = 255 * (template - min(template(:))) / (max(template(:)) - min(template(:)));
image = 255 * (image - min(image(:))) / (max(image(:)) - min(image(:)));

% Initialize warp matrix (identity transform)
init_warp = zeros(2,3);
init_warp(1,1) = 1;
init_warp(2,2) = 1;

% Test 1: Basic alignment with single level
disp('Test 1: Basic alignment (single level)')
figure('Name', 'Test 1');
[results1, results_lk1, MSE1, rho1, MSELK1] = ecc_lk(image, template, 1, 5, 'affine', init_warp);

% Test 2: With pyramid levels
disp('Test 2: Using pyramid levels')
figure('Name', 'Test 2');
[results2, results_lk2, MSE2, rho2, MSELK2] = ecc_lk(image, template, 2, 5, 'affine', init_warp);

% Test 3: With more iterations
disp('Test 3: More iterations')
figure('Name', 'Test 3');
[results3, results_lk3, MSE3, rho3, MSELK3] = ecc_lk(image, template, 1, 10, 'affine', init_warp);

% Test 4: With larger frame difference
video = VideoReader('video1_low.avi');
template = readFrame(video);
for i = 1:3 % Skip 3 frames
    image = readFrame(video);
end
image = double(image);
image = 255 * (image - min(image(:))) / (max(image(:)) - min(image(:)));

disp('Test 4: Larger frame difference')
figure('Name', 'Test 4');
[results4, results_lk4, MSE4, rho4, MSELK4] = ecc_lk(image, template, 1, 8, 'affine', init_warp);

% Create a new figure for comparison plots
figure('Name', 'Comparison of All Tests');

% Plot PSNR comparison
subplot(2,1,1);
hold on;
plot(20*log10(255./MSE1), 'b-', 'DisplayName', 'Test 1 ECC');
plot(20*log10(255./MSE2), 'r-', 'DisplayName', 'Test 2 ECC');
plot(20*log10(255./MSE3), 'g-', 'DisplayName', 'Test 3 ECC');
plot(20*log10(255./MSE4), 'k-', 'DisplayName', 'Test 4 ECC');
plot(20*log10(255./MSELK1), 'b--', 'DisplayName', 'Test 1 LK');
plot(20*log10(255./MSELK2), 'r--', 'DisplayName', 'Test 2 LK');
plot(20*log10(255./MSELK3), 'g--', 'DisplayName', 'Test 3 LK');
plot(20*log10(255./MSELK4), 'k--', 'DisplayName', 'Test 4 LK');
title('PSNR Comparison');
xlabel('Iteration');
ylabel('PSNR (dB)');
legend('Location', 'best');
grid on;

% Plot correlation coefficient comparison
subplot(2,1,2);
hold on;
plot(rho1, 'b-', 'DisplayName', 'Test 1 ECC');
plot(rho2, 'r-', 'DisplayName', 'Test 2 ECC');
plot(rho3, 'g-', 'DisplayName', 'Test 3 ECC');
plot(rho4, 'k-', 'DisplayName', 'Test 4 ECC');
title('Correlation Coefficient Comparison');
xlabel('Iteration');
ylabel('Correlation Coefficient');
legend('Location', 'best');
grid on;

% Arrange windows in a 2x2 grid
set(0,'units','pixels')  
Pix_SS = get(0,'screensize');
for i = 1:4
    figure(i)
    set(gcf, 'Position', [(mod(i-1,2))*Pix_SS(3)/2, (1-floor((i-1)/2))*Pix_SS(4)/2, ...
        Pix_SS(3)/2, Pix_SS(4)/2]);
end

% Analysis of Results
disp('Analysis of Results:')
disp('------------------')
% Compare final PSNR values for each test
disp('Final PSNR Values (Higher is better):')
disp(['Test 1 - ECC: ' num2str(20*log10(255/MSE1(end))) ' dB, LK: ' num2str(20*log10(255/MSELK1(end))) ' dB'])
disp(['Test 2 - ECC: ' num2str(20*log10(255/MSE2(end))) ' dB, LK: ' num2str(20*log10(255/MSELK2(end))) ' dB'])
disp(['Test 3 - ECC: ' num2str(20*log10(255/MSE3(end))) ' dB, LK: ' num2str(20*log10(255/MSELK3(end))) ' dB'])
disp(['Test 4 - ECC: ' num2str(20*log10(255/MSE4(end))) ' dB, LK: ' num2str(20*log10(255/MSELK4(end))) ' dB'])

% Compare correlation coefficients
disp('Final Correlation Coefficients (Closer to 1 is better):')
disp(['Test 1 - ECC: ' num2str(rho1(end)) ', LK: ' num2str(results_lk1(end).rho)])
disp(['Test 2 - ECC: ' num2str(rho2(end)) ', LK: ' num2str(results_lk2(end).rho)])
disp(['Test 3 - ECC: ' num2str(rho3(end)) ', LK: ' num2str(results_lk3(end).rho)])
disp(['Test 4 - ECC: ' num2str(rho4(end)) ', LK: ' num2str(results_lk4(end).rho)])