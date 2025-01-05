% Script for analyzing ECC and LK algorithm performance on high and low resolution sequences
clear all;
close all;

% Test configuration - modified parameters for better convergence
gap_low = 20;   % Frame gap for low resolution
gap_high = 40;  % Larger frame gap for high resolution
num_iterations = 15;      

% 1. Test with low resolution video (64x64)
disp('Testing low resolution sequence...');
video_low = VideoReader('video1_low.avi');

% Get template frame (first frame)
template_low = readFrame(video_low);
template_low = double(template_low);
template_low = (template_low - min(template_low(:))) / (max(template_low(:)) - min(template_low(:))) * 255;

% Skip frames to get a later frame
for i = 1:gap_low
    if hasFrame(video_low)
        image_low = readFrame(video_low);
    end
end
image_low = double(image_low);
image_low = (image_low - min(image_low(:))) / (max(image_low(:)) - min(image_low(:))) * 255;

% Initialize warp matrix for better conditioning
init_warp = eye(2,3);  % Identity initialization

% Use only 1 level for low resolution to avoid small image warning
num_levels_low = 1;

% Run ECC-LK alignment for low resolution
[results_low, results_lk_low, MSE_low, rho_low, MSELK_low] = ecc_lk(image_low, template_low, num_levels_low, num_iterations, 'affine', init_warp);

% 2. Test with high resolution video (256x256)
disp('Testing high resolution sequence...');
video_high = VideoReader('video1_high.avi');

% Get template frame (first frame)
template_high = readFrame(video_high);
template_high = double(template_high);
% Modified normalization for better numerical stability
template_high = (template_high - min(template_high(:)));
template_high = template_high / max(template_high(:)) * 255;

% Skip frames to get a later frame
for i = 1:gap_high
    if hasFrame(video_high)
        image_high = readFrame(video_high);
    end
end
image_high = double(image_high);
% Apply same normalization to image
image_high = (image_high - min(image_high(:)));
image_high = image_high / max(image_high(:)) * 255;

% Use single level for high resolution for better stability
num_levels_high = 1;  % Changed from 2 to 1

% Initialize warp matrix with identity for high resolution
init_warp = eye(2,3);

% Run ECC-LK alignment for high resolution
[results_high, results_lk_high, MSE_high, rho_high, MSELK_high] = ecc_lk(image_high, template_high, num_levels_high, num_iterations, 'affine', init_warp);

% Create visualization plots
figure('Name', 'Performance Comparison');

% Plot PSNR comparison
subplot(2,2,1);
hold on;
plot(20*log10(255./MSE_low), 'b-', 'DisplayName', 'Low Res ECC');
plot(20*log10(255./MSELK_low), 'b--', 'DisplayName', 'Low Res LK');
plot(20*log10(255./MSE_high), 'r-', 'DisplayName', 'High Res ECC');
plot(20*log10(255./MSELK_high), 'r--', 'DisplayName', 'High Res LK');
title('PSNR Convergence');
xlabel('Iteration');
ylabel('PSNR (dB)');
legend('Location', 'best');
grid on;

% Plot correlation coefficient comparison
subplot(2,2,2);
hold on;
plot(rho_low, 'b-', 'DisplayName', 'Low Res ECC');
plot(rho_high, 'r-', 'DisplayName', 'High Res ECC');
title('Correlation Coefficient');
xlabel('Iteration');
ylabel('Correlation');
legend('Location', 'best');
grid on;

% Display template and final warped images for low resolution
subplot(2,2,3);
montage({uint8(template_low), uint8(results_low(end,end).image)}, 'Size', [1 2]);
title('Low Resolution: Template and Final Alignment');

% Display template and final warped images for high resolution
subplot(2,2,4);
montage({uint8(template_high), uint8(results_high(end,end).image)}, 'Size', [1 2]);
title('High Resolution: Template and Final Alignment');

% Print numerical results
fprintf('\nResults Summary:\n');
fprintf('Low Resolution:\n');
fprintf('Final PSNR (ECC): %.2f dB\n', 20*log10(255/MSE_low(end)));
fprintf('Final PSNR (LK): %.2f dB\n', 20*log10(255/MSELK_low(end)));
fprintf('Final Correlation: %.4f\n', rho_low(end));

fprintf('\nHigh Resolution:\n');
fprintf('Final PSNR (ECC): %.2f dB\n', 20*log10(255/MSE_high(end)));
fprintf('Final PSNR (LK): %.2f dB\n', 20*log10(255/MSELK_high(end)));
fprintf('Final Correlation: %.4f\n', rho_high(end));