% Read the original image
img = imread('pudding.png');
[height, width, channels] = size(img);

% Define video parameters
fps = 30;                    % Frames per second
duration = 5;                % Video duration in seconds
numFrames = fps * duration;  % Total frames for 5 seconds
maxShear = 0.5;             % Maximum shear value

% Calculate required canvas size to accommodate shearing
extraWidth = round(height * maxShear * 2);  % Account for maximum shear in both directions
canvasWidth = width + extraWidth;
outputView = imref2d([height, canvasWidth]);

% Center the image in the wider canvas
outputView.XWorldLimits = [-(extraWidth/2), width + (extraWidth/2)];
outputView.YWorldLimits = [0, height];

% Create VideoWriter object
v = VideoWriter('sheared_pudding_exercise3.avi');
v.FrameRate = fps;
open(v);

% Generate frames with periodic shearing
for frame = 1:numFrames
    % Calculate shear value using sine function for smooth periodic motion
    shearAmount = maxShear * sin(2*pi*frame/(fps*duration));
    
    % Create shearing transformation matrix for horizontal shear
    tform = affine2d([1 0 0; shearAmount 1 0; 0 0 1]);
    
    % Apply shearing transformation with wider output view
    shearedImg = imwarp(img, tform, 'OutputView', outputView);
    
    % Write frame to video
    writeVideo(v, shearedImg);
end

% Close the video file
close(v);