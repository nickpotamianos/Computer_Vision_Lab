function SIFT_RANSAC()
% SIFT_RANSAC.m
% -------------------------
% This script demonstrates:
%   1) Reading two images
%   2) Detecting SIFT features & computing descriptors (using your professor's code)
%   3) Matching descriptors (ratio test)
%   4) Estimating a homography using RANSAC
%   5) Displaying inlier matches
%
% INSTRUCTIONS:
%  1) Place this file and SIFT_feature.m in the same folder.
%  2) Put your images (e.g., 'image1.jpg' and 'image2.jpg') in the same folder (or provide correct path).
%  3) In MATLAB Command Window, run: 
%       >> SIFT_RANSAC
%
%   You should see a figure of the inlier matches after RANSAC.

    close all; clc;
    
    %-------------------------
    % 1) Read images
    %-------------------------
    I1 = imread('cameraman.tif');  % <-- change file name as needed
    I2 = imread('cameraman.tif');  % <-- change file name as needed

    % Convert to grayscale if needed
    if size(I1,3) == 3
        I1_gray = rgb2gray(I1);
    else
        I1_gray = I1;
    end

    if size(I2,3) == 3
        I2_gray = rgb2gray(I2);
    else
        I2_gray = I2;
    end

    %-------------------------
    % 2) Extract SIFT features & descriptors
    %    (Calls your professor's code; adapt if needed)
    %-------------------------
    [kp1, desc1] = SIFT_feature_wrapper(I1_gray);
    [kp2, desc2] = SIFT_feature_wrapper(I2_gray);

    %-------------------------
    % 3) Match descriptors (ratio test)
    %-------------------------
    ratioThreshold = 0.75; 
    matches = siftMatch(desc1, desc2, ratioThreshold);
    % matches is Nx2, each row = [indexInDesc1, indexInDesc2]

    % Extract matched (x,y) positions from kp1, kp2
    matchedPoints1 = kp1(matches(:,1), :);  % Nx2
    matchedPoints2 = kp2(matches(:,2), :);  % Nx2

    %-------------------------
    % 4) RANSAC to find homography
    %-------------------------
    ransacThreshold = 5;     % allowable reprojection error in pixels
    maxIterations   = 2000;  % number of RANSAC iterations

    [bestH, inlierIndices] = ransacHomography(matchedPoints1, matchedPoints2, ...
                                              ransacThreshold, maxIterations);

    inliers1 = matchedPoints1(inlierIndices,:);
    inliers2 = matchedPoints2(inlierIndices,:);
    fprintf('Number of inliers: %d / %d total matches\n', length(inlierIndices), size(matches,1));

    %-------------------------
    % 5) Display inlier matches
    %-------------------------
    figure;
    % Requires Computer Vision Toolbox:
    showMatchedFeatures(I1, I2, inliers1, inliers2, 'montage');
    title('Inlier Matches After RANSAC');
end

% ========================================================================
% SUBFUNCTIONS
% ========================================================================

function [kp, desc] = SIFT_feature_wrapper(img_gray)
% SIFT_feature_wrapper
%   Calls your professor's SIFT_feature.m code, which should produce
%   something like 'feature' (128xN) and 'extrema' (or 'keypoints').
%
%   Here we assume SIFT_feature.m returns [features, keypoints] with:
%     features = 128 x N  (each column is a descriptor)
%     keypoints = Nx2 or Nx4 or something similar
%
%   You must adapt to how your own SIFT_feature.m is structured.

    % Call your professor's function. 
    % In many distributions, SIFT_feature might be a SCRIPT. If it's truly
    % a function, you can do:
    [features, keypoints] = SIFT_feature(img_gray);

    % If your professor's code outputs keypoints in (row, col) format, you might
    % want to invert them so that they are (x, y). Here, we assume they are
    % already in [x, y]. If not, swap columns.

    % 'features' is 128 x N => transpose to Nx128 for easier handling
    desc = features';  % Nx128
    kp   = keypoints;  % Nx2  (assuming already [x, y])

    % If your keypoints are Nx4, you might need:
    % kp = keypoints(:,1:2); 
end

% ------------------------------------------------------------------------
function matches = siftMatch(desc1, desc2, ratio)
% siftMatch
%   Brute-force match SIFT descriptors in desc1 to desc2 using Lowe's
%   ratio test. Typically ratio ~ 0.7-0.8.
%
%   desc1: Nx128
%   desc2: Mx128
%   matches: Kx2 array of matched indices: [idxInDesc1, idxInDesc2]

    matches = [];
    for i = 1:size(desc1,1)
        d1 = desc1(i,:);
        
        % Euclidian distances to all descriptors in desc2
        distances = sum((desc2 - d1).^2, 2);  % Mx1

        [sortedDist, sortedIdx] = sort(distances, 'ascend');
        bestDist    = sortedDist(1);
        bestIdx     = sortedIdx(1);
        secondDist  = sortedDist(2);

        % Ratio test (compare squared distances)
        if bestDist < ratio^2 * secondDist
            matches = [matches; [i, bestIdx]];
        end
    end
end

% ------------------------------------------------------------------------
function [bestH, bestInlierIdx] = ransacHomography(pts1, pts2, threshold, maxIter)
% ransacHomography
%   Estimate homography using RANSAC.
%   pts1, pts2: Nx2 matched coordinates
%   threshold: inlier threshold (pixels)
%   maxIter: number of iterations
%
%   Returns:
%     bestH         - the 3x3 homography
%     bestInlierIdx - indices of inlier matches

    numPoints = size(pts1,1);
    bestH = eye(3);
    bestInlierIdx = [];
    maxInliers = 0;

    for i = 1:maxIter
        % Randomly pick 4 distinct points
        randIdx = randperm(numPoints, 4);
        sample1 = pts1(randIdx, :);
        sample2 = pts2(randIdx, :);

        % Compute homography from these 4 correspondences
        H = estimateHomography(sample1, sample2);

        % Project all pts1 -> pts2
        projectedPts2 = applyHomography(pts1, H);
        dists = sqrt(sum((projectedPts2 - pts2).^2, 2));  % Nx1

        inliers = find(dists < threshold);
        numInliers = length(inliers);

        if numInliers > maxInliers
            maxInliers = numInliers;
            bestInlierIdx = inliers;
            bestH = H;
        end
    end

    % (Optional) refine using all inliers:
    if ~isempty(bestInlierIdx)
        bestH = estimateHomography(pts1(bestInlierIdx,:), pts2(bestInlierIdx,:));
    end
end

% ------------------------------------------------------------------------
function H = estimateHomography(pts1, pts2)
% estimateHomography 
%   Solve for 3x3 H that maps pts1 -> pts2.
%   pts1, pts2: Nx2
%   Use DLT (Direct Linear Transform).

    N = size(pts1, 1);
    A = zeros(2*N, 9);

    for i = 1:N
        x1 = pts1(i,1); 
        y1 = pts1(i,2);
        x2 = pts2(i,1);
        y2 = pts2(i,2);

        A(2*i-1,:) = [ x1, y1, 1,   0,  0,  0,  -x2*x1,  -x2*y1,  -x2 ];
        A(2*i,  :) = [ 0,  0,  0,  x1, y1, 1,  -y2*x1,  -y2*y1,  -y2 ];
    end

    [~,~,V] = svd(A);
    h = V(:,end);    % last column
    H = reshape(h, [3,3])';
    H = H ./ H(3,3); % normalize so H(3,3) = 1
end

% ------------------------------------------------------------------------
function pts2 = applyHomography(pts1, H)
% applyHomography
%   pts1: Nx2
%   H: 3x3
%   returns pts2: Nx2

    N = size(pts1,1);
    pts1_hom = [pts1, ones(N,1)]';    % 3xN
    pts2_hom = H * pts1_hom;         % 3xN
    pts2_hom = pts2_hom ./ pts2_hom(3,:); % normalize
    pts2 = pts2_hom(1:2,:)';         % Nx2
end
