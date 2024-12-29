% Read the original image
img = imread('pudding.png');
[height, width, channels] = size(img);

% Define video parameters
fps = 30;           % Frames per second
duration = 5;       % Duration in seconds (increased to 5)
numFrames = fps * duration;
maxShear = 0.5;     % Maximum shear value

% Calculate canvas size needed for shearing
extraWidth = round(width * maxShear * 2);  % Double the extra width to accommodate both sides
canvasWidth = width + extraWidth;

% Create VideoWriter object
v = VideoWriter('sheared_pudding.avi');
v.FrameRate = fps;
open(v);

% Create reference object for output image
outputView = imref2d([height, canvasWidth]);

% Center the image in the canvas
outputView.XWorldLimits = [-extraWidth/2, width + extraWidth/2];
outputView.YWorldLimits = [0, height];

% Generate frames with periodic shearing
for frame = 1:numFrames
    % Calculate shear value using sine function for smooth periodic motion
    shearAmount = maxShear * sin(2*pi*frame/numFrames);  % Changed to sin to oscillate around center
    
    % Create affine transformation matrix for horizontal shear
    % Add translation to keep bottom fixed
    tform = affine2d([1 0 0; shearAmount 1 0; -shearAmount*height 0 1]);
    
    % Apply transformation
    shearedImg = imwarp(img, tform, 'OutputView', outputView);
    
    % Write frame to video
    writeVideo(v, shearedImg);
end

% Close the video file
close(v);