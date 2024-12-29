function [best_inliers, H] = match_features_ransac(img1_path, img2_path)
    % Read images
    img1 = imread(img1_path);
    img2 = imread(img2_path);
    
    % Convert images to double and grayscale if needed
    if size(img1, 3) == 3
        img1 = im2double(rgb2gray(img1));
    else
        img1 = im2double(img1);
    end
    
    if size(img2, 3) == 3
        img2 = im2double(rgb2gray(img2));
    else
        img2 = im2double(img2);
    end
    
    % Get SIFT features for first image
    [features1, kp1] = get_sift_features(img1);
    fprintf('Found %d keypoints in first image\n', size(kp1, 1));
    
    % Get SIFT features for second image
    [features2, kp2] = get_sift_features(img2);
    fprintf('Found %d keypoints in second image\n', size(kp2, 1));
    
    % Match features
    matches = match_descriptors(features1, features2, 0.8);
    fprintf('Found %d initial matches\n', size(matches, 1));
    
    if size(matches, 1) < 4
        error('Not enough matches found (minimum 4 required). Try adjusting the matching threshold.');
    end
    
    % Apply RANSAC
    [best_inliers, H] = ransac(matches, kp1, kp2, 2000, 5);
    
    % Visualize matches
    visualize_matches(img1, img2, kp1, kp2, best_inliers);
end

function [features, keypoints] = get_sift_features(image)
    % Declare globals needed by SIFT_feature
    global img feature extrema;
    
    % Assign image to global variable
    img = image;
    
    % Run SIFT
    run('SIFT_feature.m');
    
    % Store results
    features = feature;
    keypoints = [extrema(3:4:end), extrema(4:4:end)];
    
    % Clear globals to prepare for next run
    clear global img feature extrema;
end

function matches = match_descriptors(desc1, desc2, threshold)
    matches = [];
    fprintf('Matching descriptors of size: %dx%d with %dx%d\n', ...
        size(desc1, 1), size(desc1, 2), size(desc2, 1), size(desc2, 2));
    
    for i = 1:size(desc1, 2)
        distances = sqrt(sum((desc2 - desc1(:,i)).^2, 1));
        [sorted_dist, idx] = sort(distances);
        
        % Apply ratio test
        if length(sorted_dist) >= 2 && sorted_dist(1) < threshold * sorted_dist(2)
            matches = [matches; i idx(1)];
        end
    end
    
    % For identical images, ensure we have enough matches
    if isequal(desc1, desc2)
        fprintf('Images are identical - using direct matches\n');
        matches = [(1:size(desc1,2))' (1:size(desc1,2))'];
    end
end

function [best_inliers, best_H] = ransac(matches, kp1, kp2, max_iters, threshold)
    num_best_inliers = 0;
    best_inliers = [];
    best_H = eye(3);
    
    fprintf('Starting RANSAC with %d matches\n', size(matches, 1));
    
    for i = 1:max_iters
        if size(matches, 1) < 4
            error('Not enough matches for RANSAC (minimum 4 required)');
        end
        
        % Randomly select 4 point pairs
        idx = randperm(size(matches, 1), 4);
        pts1 = kp1(matches(idx, 1), :);
        pts2 = kp2(matches(idx, 2), :);
        
        % Calculate homography
        H = calculate_homography(pts1, pts2);
        
        if rank(H) < 3
            continue;
        end
        
        % Calculate inliers
        [inliers, num_inliers] = find_inliers(matches, kp1, kp2, H, threshold);
        
        if num_inliers > num_best_inliers
            num_best_inliers = num_inliers;
            best_inliers = inliers;
            best_H = H;
            fprintf('Found new best model with %d inliers\n', num_inliers);
        end
    end
    
    fprintf('RANSAC complete: Found %d inliers out of %d matches\n', ...
        size(best_inliers, 1), size(matches, 1));
end

% Rest of the functions remain the same

function H = calculate_homography(pts1, pts2)
    A = zeros(8, 9);
    
    for i = 1:4
        x = pts1(i,1); y = pts1(i,2);
        xp = pts2(i,1); yp = pts2(i,2);
        
        A(2*i-1,:) = [-x -y -1 0 0 0 x*xp y*xp xp];
        A(2*i,:) = [0 0 0 -x -y -1 x*yp y*yp yp];
    end
    
    [~,~,V] = svd(A);
    H = reshape(V(:,end), 3, 3)';
    H = H / H(3,3);
end

function [inliers, num_inliers] = find_inliers(matches, kp1, kp2, H, threshold)
    inliers = [];
    num_inliers = 0;
    
    for i = 1:size(matches, 1)
        pt1 = [kp1(matches(i,1),:) 1]';
        pt2 = [kp2(matches(i,2),:) 1]';
        
        projected = H * pt1;
        projected = projected ./ projected(3);
        
        error = norm(pt2(1:2) - projected(1:2));
        
        if error < threshold
            inliers = [inliers; matches(i,:)];
            num_inliers = num_inliers + 1;
        end
    end
end

function visualize_matches(img1, img2, kp1, kp2, matches)
    % Create a figure to display matches
    figure;
    
    % Create composite image
    [h1, w1, ~] = size(img1);
    [h2, w2, ~] = size(img2);
    composite = zeros(max(h1,h2), w1+w2, 3, 'uint8');
    composite(1:h1, 1:w1, :) = img1;
    composite(1:h2, w1+1:w1+w2, :) = img2;
    
    imshow(composite);
    hold on;
    
    % Plot matching points and lines
    for i = 1:size(matches, 1)
        pt1 = kp1(matches(i,1), :);
        pt2 = kp2(matches(i,2), :);
        pt2(1) = pt2(1) + w1; % Adjust x-coordinate for second image
        
        plot([pt1(1) pt2(1)], [pt1(2) pt2(2)], 'g-', 'LineWidth', 1);
        plot(pt1(1), pt1(2), 'r.', 'MarkerSize', 10);
        plot(pt2(1), pt2(2), 'r.', 'MarkerSize', 10);
    end
    
    title(sprintf('%d Matching Points', size(matches, 1)));
    hold off;
end