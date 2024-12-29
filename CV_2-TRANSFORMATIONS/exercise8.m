function create_beach_ball_horizon_animation()
    try
        % Read images
        ball = imread('ball.jpg');
        mask = imread('ball_mask.jpg');
        background = imread('beach.jpg');

        % Convert images to double for processing
        ball = im2double(ball);
        mask = im2double(rgb2gray(mask));
        background = im2double(background);

        % Invert the mask
        mask = ~mask;

        % Resize background to standard size
        background = imresize(background, [720 1280]);

        % Initial ball size and resize
        ball_size = 100;
        ball = imresize(ball, [ball_size ball_size]);
        mask = imresize(mask, [ball_size ball_size]);

        % Get dimensions
        [height, width, ~] = size(background);

        % Create video writer
        v = VideoWriter('transf_beach.avi');
        v.FrameRate = 30;
        open(v);

        % Animation parameters
        num_frames = 150;

        % Create time vector
        t = linspace(0, 1, num_frames);

        % Starting positions
        start_x = width / 4;
        start_y = height / 2;

        % Calculate trajectories
        x_positions = linspace(start_x, width * 3/4, num_frames);
        y_positions = start_y - 200 * sin(pi * t);

        % Scale factors (ball gets smaller as it moves)
        scale_factors = linspace(1.0, 0.3, num_frames);

        % Rotation angles
        rotation_angles = linspace(0, 360, num_frames);

        % Main animation loop
        for frame = 1:num_frames
            % Get current position and parameters
            current_x = round(x_positions(frame));
            current_y = round(y_positions(frame));
            scale = scale_factors(frame);
            angle = rotation_angles(frame);

            % Scale ball and mask
            current_size = max(1, round(ball_size * scale));
            scaled_ball = imresize(ball, [current_size current_size]);
            scaled_mask = imresize(mask, [current_size current_size]);

            % Rotate ball and mask using bilinear interpolation
            rotated_ball = imrotate(scaled_ball, angle, 'bilinear', 'crop');
            rotated_mask = imrotate(scaled_mask, angle, 'bilinear', 'crop');

            % Create frame
            current_frame = background;

            % Calculate ROI coordinates
            y1 = max(1, round(current_y - current_size/2));
            y2 = min(height, y1 + current_size - 1);
            x1 = max(1, round(current_x - current_size/2));
            x2 = min(width, x1 + current_size - 1);

            % Calculate valid regions
            roi_height = y2 - y1 + 1;
            roi_width = x2 - x1 + 1;

            % Get regions from rotated images
            ball_roi = rotated_ball(1:roi_height, 1:roi_width, :);
            mask_roi = rotated_mask(1:roi_height, 1:roi_width);

            % Get ROI from current frame
            roi = current_frame(y1:y2, x1:x2, :);

            % Apply mask to each channel
            for c = 1:3
                roi(:,:,c) = ball_roi(:,:,c) .* mask_roi + roi(:,:,c) .* (1-mask_roi);
            end

            % Insert ROI back into frame
            current_frame(y1:y2, x1:x2, :) = roi;

            % Convert to uint8 before writing
            current_frame = im2uint8(current_frame);

            % Write frame
            writeVideo(v, current_frame);
        end

        % Close video writer
        close(v);
        disp('Beach ball horizon animation created successfully!');

    catch ME
        disp(['An error occurred: ' ME.message]);
        if exist('v', 'var') && isopen(v)
            close(v);
        end
    end
end

% Run the function
create_beach_ball_horizon_animation();