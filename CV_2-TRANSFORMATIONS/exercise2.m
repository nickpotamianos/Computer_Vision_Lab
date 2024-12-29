% Read the original image
img = imread('pudding.png');
[height, width, channels] = size(img);

% Create a wider canvas to hold the horizontal composition
canvas_width = width * 3;  % Wide enough for 5 puddings
canvas_height = height;    % Original height
canvas = uint8(zeros(canvas_height, canvas_width, channels));

% Define scaling factors for 5 puddings
scales = linspace(0.2, 1.0, 5);

% Calculate adaptive spacing based on pudding sizes
total_scaled_width = 0;
scaled_widths = [];
for i = 1:5
    scaled_w = round(width * scales(i));
    scaled_widths(i) = scaled_w;
    total_scaled_width = total_scaled_width + scaled_w;
end

% Calculate total spacing available
total_space = canvas_width - total_scaled_width;
% Space between puddings will be proportional to their average size
space_unit = total_space / 16;  % 4 gaps between 5 puddings

% Calculate positions
positions = zeros(1, 5);
current_pos = 1;
for i = 1:5
    positions(i) = current_pos + scaled_widths(i)/2;
    if i < 5
        % Add space proportional to the average size of current and next pudding
        current_pos = current_pos + scaled_widths(i) + space_unit * (scales(i) + scales(i+1))/2;
    end
end

transformations = {};
for i = 1:5
    % [scale_x, scale_y, pos_x, pos_y]
    transformations{i} = [scales(i), scales(i), ...
        positions(i), ...     % calculated horizontal positions
        scaled_widths(i)/2];  % align to top by using half height of scaled pudding
end

% Apply each transformation and compose the image
for i = 1:length(transformations)
    scale_x = transformations{i}(1);
    scale_y = transformations{i}(2);
    pos_x = transformations{i}(3);
    pos_y = transformations{i}(4);
    
    % Create scaling transform
    tform = affine2d([scale_x 0 0; 0 scale_y 0; 0 0 1]);
    
    % Create reference object for proper positioning
    ref = imref2d([round(height*scale_y), round(width*scale_x)]);
    
    % Apply transformation
    scaled_img = imwarp(img, tform, 'OutputView', ref);
    
    % Calculate position to place the scaled image
    start_y = max(1, round(pos_y - (size(scaled_img,1)/2)));
    end_y = min(canvas_height, start_y + size(scaled_img,1) - 1);
    start_x = max(1, round(pos_x - (size(scaled_img,2)/2)));
    end_x = min(canvas_width, start_x + size(scaled_img,2) - 1);
    
    % Adjust scaled_img indices if necessary
    img_start_y = 1;
    img_start_x = 1;
    if start_y < 1
        img_start_y = 2 - start_y;
        start_y = 1;
    end
    if start_x < 1
        img_start_x = 2 - start_x;
        start_x = 1;
    end
    
    % Place the scaled image on canvas
    canvas(start_y:end_y, start_x:end_x, :) = scaled_img(img_start_y:img_start_y+end_y-start_y, ...
        img_start_x:img_start_x+end_x-start_x, :);
end

% Display the result
figure;
imshow(canvas);
title('Puddings in Ascending Size');

% Save the result
imwrite(canvas, 'scaled_pudding_line.png');