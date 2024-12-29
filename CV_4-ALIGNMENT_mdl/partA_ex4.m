% Script to analyze sequential frame alignment performance
clear all;
close all;

% Configuration
num_iterations = 15;
num_levels = 1;
init_warp = eye(2,3);  % Identity initialization

% Define video files to process
video_files = {'video1_low.avi', 'video1_high.avi', 'video2_low.avi', 'video2_high.avi'};
video_names = {'Video 1 Low Res', 'Video 1 High Res', 'Video 2 Low Res', 'Video 2 High Res'};

% Store all PSNR results
all_psnr_ecc = cell(1,4);
all_psnr_lk = cell(1,4);
all_frame_pairs = cell(1,4);

% First check if files exist
fprintf('\nChecking video files:\n');
for i = 1:length(video_files)
    if exist(video_files{i}, 'file') == 2
        fprintf('%s found\n', video_files{i});
    else
        fprintf('WARNING: %s not found in current directory!\n', video_files{i});
        fprintf('Current directory is: %s\n', pwd);
    end
end

% Process each video and store results
for vid_idx = 1:length(video_files)
    try
        % Check if file exists
        if ~exist(video_files{vid_idx}, 'file')
            error('Video file %s not found', video_files{vid_idx});
        end
        
        % Try to load video
        disp(['\nProcessing ' video_names{vid_idx}]);
        video = VideoReader(video_files{vid_idx});
        fprintf('Successfully loaded %s\n', video_files{vid_idx});
        fprintf('Video properties: %d x %d, %d frames\n', ...
            video.Width, video.Height, video.NumFrames);
        
        % Arrays to store results
        psnr_ecc = [];
        psnr_lk = [];
        frame_pairs = [];
        
        % Get first frame as template
        if hasFrame(video)
            template = readFrame(video);
            if size(template, 3) > 1
                template = rgb2gray(template);
            end
            template = double(template);
            template = (template - min(template(:))) / (max(template(:)) - min(template(:))) * 255;
            
            frame_count = 1;
            fprintf('Processing frames: ');
            
            % Process consecutive frames
            while hasFrame(video)
                if mod(frame_count, 10) == 0
                    fprintf('%d ', frame_count);
                end
                
                % Get next frame
                image = readFrame(video);
                if size(image, 3) > 1
                    image = rgb2gray(image);
                end
                image = double(image);
                image = (image - min(image(:))) / (max(image(:)) - min(image(:))) * 255;
                
                % Run alignment with plot_flag=0 to prevent intermediate plots
                [results, results_lk, MSE, rho, MSELK] = ecc_lk(image, template, num_levels, num_iterations, 'affine', init_warp);
                
                % Store results
                psnr_ecc(end+1) = 20*log10(255/sqrt(MSE(end)));
                psnr_lk(end+1) = 20*log10(255/sqrt(MSELK(end)));
                frame_pairs(end+1) = frame_count;
                
                % Update template for next iteration
                template = image;
                frame_count = frame_count + 1;
                
                % Close any figures created by ecc_lk
                close all;
            end
            fprintf('\nProcessed %d frames\n', frame_count-1);
            
            % Store results for this video
            all_psnr_ecc{vid_idx} = psnr_ecc;
            all_psnr_lk{vid_idx} = psnr_lk;
            all_frame_pairs{vid_idx} = frame_pairs;
            
            % Print statistics
            fprintf('\nResults for %s:\n', video_names{vid_idx});
            fprintf('Average PSNR (ECC): %.2f dB\n', mean(psnr_ecc));
            fprintf('Average PSNR (LK): %.2f dB\n', mean(psnr_lk));
            
        else
            error('No frames in video');
        end
        
    catch e
        fprintf('\nError processing %s:\n%s\n', video_files{vid_idx}, e.message);
        all_psnr_ecc{vid_idx} = [];
        all_psnr_lk{vid_idx} = [];
        all_frame_pairs{vid_idx} = [];
    end
end

% Create new figure for final plots
figure('Name', 'Sequential Frame Alignment Analysis');
set(gcf, 'Color', 'white');
set(gcf, 'Position', [100 100 1200 800]);  % Make figure larger

% Define colors and styles for better visibility
colors_ecc = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880]};  % Blue, Orange, Purple, Green
colors_lk = {[0.3010 0.7450 0.9330], [0.9290 0.6940 0.1250], [0.6350 0.0780 0.1840], [0.9290 0.6940 0.1250]}; % Light Blue, Yellow, Dark Red, Yellow

% Create subplots for each resolution
subplot(2,1,1)  % Low resolution videos
hold on;
box on;
for i = [1,3]  % Low res videos
    if ~isempty(all_psnr_ecc{i})
        % Plot ECC with solid thick line
        plot(all_frame_pairs{i}, all_psnr_ecc{i}, 'Color', colors_ecc{i}, 'LineWidth', 2, 'LineStyle', '-', ...
            'DisplayName', [video_names{i} ' ECC']);
        % Plot LK with dashed thinner line
        plot(all_frame_pairs{i}, all_psnr_lk{i}, 'Color', colors_lk{i}, 'LineWidth', 1.5, 'LineStyle', '--', ...
            'DisplayName', [video_names{i} ' LK']);
    end
end
title('Low Resolution Videos', 'FontSize', 12);
xlabel('Frame Pair', 'FontSize', 11);
ylabel('PSNR (dB)', 'FontSize', 11);
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'GridAlpha', 0.2);
ylim([min([cell2mat(all_psnr_ecc([1,3])) cell2mat(all_psnr_lk([1,3]))]) - 0.5, ...
      max([cell2mat(all_psnr_ecc([1,3])) cell2mat(all_psnr_lk([1,3]))]) + 0.5]);

subplot(2,1,2)  % High resolution videos
hold on;
box on;
for i = [2,4]  % High res videos
    if ~isempty(all_psnr_ecc{i})
        % Plot ECC with solid thick line
        plot(all_frame_pairs{i}, all_psnr_ecc{i}, 'Color', colors_ecc{i}, 'LineWidth', 2, 'LineStyle', '-', ...
            'DisplayName', [video_names{i} ' ECC']);
        % Plot LK with dashed thinner line
        plot(all_frame_pairs{i}, all_psnr_lk{i}, 'Color', colors_lk{i}, 'LineWidth', 1.5, 'LineStyle', '--', ...
            'DisplayName', [video_names{i} ' LK']);
    end
end
title('High Resolution Videos', 'FontSize', 12);
xlabel('Frame Pair', 'FontSize', 11);
ylabel('PSNR (dB)', 'FontSize', 11);
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'GridAlpha', 0.2);
ylim([min([cell2mat(all_psnr_ecc([2,4])) cell2mat(all_psnr_lk([2,4]))]) - 0.5, ...
      max([cell2mat(all_psnr_ecc([2,4])) cell2mat(all_psnr_lk([2,4]))]) + 0.5]);

% Adjust overall figure layout
sgtitle('PSNR Performance for Sequential Frame Alignment', 'FontSize', 14);