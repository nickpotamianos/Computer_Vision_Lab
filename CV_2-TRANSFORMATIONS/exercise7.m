function create_beach_ball_animation()
    
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

        % Resize images
        background = imresize(background, [720 1280]);
        ball_size = 80;
        ball = imresize(ball, [ball_size ball_size]);
        mask = imresize(mask, [ball_size ball_size]);

        % Get dimensions
        [height, width, ~] = size(background);

        % Create video writer
        v = VideoWriter('transf_beach_ex7.avi');
        v.FrameRate = 30;
        open(v);

        % Animation parameters
        num_frames = 120;

        % Bouncing parameters
        gravity = 0.5;
        bounce_height = 200;
        horizontal_speed = 8;

        % Starting position
        x = width / 4;
        y = height - ball_size - 50;  % Start near bottom of frame

        % Initial vertical velocity (negative means going up)
        velocity = -20;

        % Rotation angle
        angle = 0;

        % Main animation loop
        for frame = 1:num_frames
            % Update position
            x = x + horizontal_speed;
            velocity = velocity + gravity;
            y = y + velocity;

            % Bounce when hitting the ground
            if y > height - ball_size - 50  % Ground position
                y = height - ball_size - 50;
                velocity = -abs(velocity * 0.7);  % Reduce bounce height by 30%
            end

            % Update rotation
            angle = angle + 5;  % Rotate 5 degrees per frame

            % Rotate ball and mask using bilinear interpolation
            rotated_ball = imrotate(ball, angle, 'bilinear', 'crop');
            rotated_mask = imrotate(mask, angle, 'bilinear', 'crop');

            % Create frame
            current_frame = background;

            % Calculate ROI coordinates
            y1 = max(1, round(y - ball_size/2));
            y2 = min(height, y1 + ball_size - 1);
            x1 = max(1, round(x - ball_size/2));
            x2 = min(width, x1 + ball_size - 1);

            % Make sure we don't exceed bounds
            roi_height = y2 - y1 + 1;
            roi_width = x2 - x1 + 1;
            
            % Get corresponding regions from rotated images
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

            % Reset ball position if it goes off screen
            if x > width - ball_size
                x = ball_size;
            end
        end

        % Close video writer
        close(v);
        disp('Beach ball animation created successfully!');


end

% Run the function
create_beach_ball_animation();