% Read images
windmill = imread('windmill.png');
mask = imread('windmill_mask.png');
background = imread('windmill_back.jpeg');

% Convert images to double for processing
windmill = im2double(windmill);
mask = im2double(rgb2gray(mask));
background = im2double(background);

% Invert the mask
mask = ~mask;

% Find bounding box
[row,col] = find(mask);
crop_top = min(row);
crop_bottom = max(row);
crop_left = min(col);
crop_right = max(col);

% Crop images
mask = mask(crop_top:crop_bottom, crop_left:crop_right);
windmill = windmill(crop_top:crop_bottom, crop_left:crop_right, :);

% Resize
new_width = round(size(mask,2)/2);
new_height = round(size(mask,1)/2);
mask = imresize(mask, [new_height new_width]);
windmill = imresize(windmill, [new_height new_width]);
background = imresize(background, 0.5);

% Get dimensions
[bg_height, bg_width, ~] = size(background);
y_offset = round((bg_height - new_height)/2);
x_offset = round((bg_width - new_width)/2);

% Create video writer
v = VideoWriter('transf_windmill_matlab.avi');
v.FrameRate = 30;
open(v);

% Animation parameters
num_frames = 200;
angle_step = -360/num_frames;

% Main loop
for frame = 1:num_frames
    % Calculate rotation angle
    angle = (frame-1) * angle_step;
    
    % Create rotation matrix
    R = [cosd(angle) -sind(angle); sind(angle) cosd(angle)];
    
    % Rotate windmill and mask
    rotated_windmill = imrotate(windmill, angle, 'bilinear', 'crop');
    rotated_mask = imrotate(mask, angle, 'bilinear', 'crop');
    
    % Create frame
    current_frame = background;
    
    % Get ROI
    roi = current_frame(y_offset+1:y_offset+new_height, x_offset+1:x_offset+new_width, :);
    
    % Apply mask
    for c = 1:3
        roi(:,:,c) = rotated_windmill(:,:,c) .* rotated_mask + roi(:,:,c) .* (1-rotated_mask);
    end
    
    % Insert ROI back
    current_frame(y_offset+1:y_offset+new_height, x_offset+1:x_offset+new_width, :) = roi;
    
    % Convert to uint8 before writing
    current_frame = im2uint8(current_frame);
    
    % Write frame
    writeVideo(v, current_frame);
end

close(v);