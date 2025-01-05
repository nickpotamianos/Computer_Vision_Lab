% match_SIFT_RANSAC.m
% 
% Αυτό το σκριπτ υλοποιεί την αντιστοίχιση σημείων-κλειδιών μεταξύ δύο εικόνων
% χρησιμοποιώντας τον αλγόριθμο SIFT και RANSAC για την απομάκρυνση ακατάλληλων αντιστοιχίσεων.
%
% Προαπαιτούμενα:
% - SIFT_feature_func.m πρέπει να βρίσκεται στον ίδιο φάκελο με αυτό το σκριπτ.
% - Εγκατεστημένο το Computer Vision Toolbox και άλλα απαιτούμενα toolboxes.
%
% Χρήση:
% - Αντικαταστήστε τα 'image1.jpg' και 'image2.jpg' με τα ονόματα των εικόνων σας.
%
% Συγγραφέας: [Το Όνομά Σας]
% Ημερομηνία: [Ημερομηνία]

%% Αρχικοποίηση
clear; clc; close all;

%% Ορισμός Ονομάτων Εικόνων
image1_filename = 'cameraman_zoom.png'; % Αντικαταστήστε με το όνομα της πρώτης εικόνας
image2_filename = 'cameraman_zoom.png'; % Αντικαταστήστε με το όνομα της δεύτερης εικόνας

%% Συνάρτηση για Επεξεργασία Εικόνας με SIFT_feature_func.m
% Οι περιγραφείς και τα σημεία-κλειδιά αποθηκεύονται στις μεταβλητές 'features', 'keypoints_x', 'keypoints_y'.

%% Επεξεργασία Πρώτης Εικόνας
fprintf('Επεξεργασία πρώτης εικόνας: %s\n', image1_filename);
% Διαβάστε την πρώτη εικόνα
img1 = imread(image1_filename);

% Μετατροπή σε grayscale αν είναι χρωματική
if size(img1,3) == 3
    img1 = rgb2gray(img1);
end

% Αλλαγή μεγέθους σε 256x256 αν χρειάζεται
img1 = imresize(img1, [256, 256]);

% Μετατροπή σε double
img1 = im2double(img1);

% Εκτέλεση του SIFT_feature_func.m
[features1, keypoints1_x, keypoints1_y] = SIFT_feature_func(img1);

fprintf('Αριθμός σημείων-κλειδιών στην πρώτη εικόνα: %d\n', length(keypoints1_x));

%% Επεξεργασία Δεύτερης Εικόνας
fprintf('Επεξεργασία δεύτερης εικόνας: %s\n', image2_filename);
% Διαβάστε τη δεύτερη εικόνα
img2 = imread(image2_filename);

% Μετατροπή σε grayscale αν είναι χρωματική
if size(img2,3) == 3
    img2 = rgb2gray(img2);
end

% Αλλαγή μεγέθους σε 256x256 αν χρειάζεται
img2 = imresize(img2, [256, 256]);

% Μετατροπή σε double
img2 = im2double(img2);

% Εκτέλεση του SIFT_feature_func.m
[features2, keypoints2_x, keypoints2_y] = SIFT_feature_func(img2);

fprintf('Αριθμός σημείων-κλειδιών στη δεύτερη εικόνα: %d\n', length(keypoints2_x));

%% Αντιστοίχιση Περιγραφέων με Χρήση Ratio Test
fprintf('Αντιστοίχιση περιγραφέων με χρήση Ratio Test...\n');

% Μετατροπή περιγραφέων σε μορφή n x 128
features1_t = double(features1');
features2_t = double(features2');

% Υπολογισμός αποστάσεων Euclidean μεταξύ όλων των ζευγών περιγραφέων
distances = pdist2(features1_t, features2_t, 'euclidean');

% Ταξινόμηση αποστάσεων για κάθε περιγραφέα της πρώτης εικόνας
[sorted_distances, sorted_indices] = sort(distances, 2);

% Εφαρμογή του Ratio Test του Lowe (π.χ., ratio = 0.8)
ratio_thresh = 0.8;
matches = sorted_distances(:,1) < ratio_thresh * sorted_distances(:,2);

% Βρες τους δείκτες των αντιστοιχισμένων περιγραφέων
matched_indices1 = find(matches);
matched_indices2 = sorted_indices(matches,1);

% Βεβαιωθείτε ότι υπάρχουν αντιστοιχίσεις
if isempty(matched_indices1)
    error('Δεν βρέθηκαν αντιστοιχίσεις μετά το Ratio Test.');
end

% Αποθήκευση των αντιστοιχισμένων σημείων-κλειδιών
matched_points1 = [keypoints1_x(matched_indices1)', keypoints1_y(matched_indices1)'];
matched_points2 = [keypoints2_x(matched_indices2)', keypoints2_y(matched_indices2)'];

fprintf('Αριθμός αντιστοιχιών μετά το Ratio Test: %d\n', size(matched_points1,1));

%% Εφαρμογή του Αλγορίθμου RANSAC για Βελτίωση Αντιστοιχιών
fprintf('Εφαρμογή του αλγορίθμου RANSAC για εύρεση inliers...\n');

% Ορισμός παραμέτρων RANSAC
ransac_threshold = 3; % Όριο αποστάσεων σε pixels
ransac_max_iters = 1000; % Μέγιστος αριθμός επαναλήψεων

% Αρχικοποίηση μεταβλητών
best_inliers = [];
best_H = eye(3);

% Μετατροπή σημείων σε ομογενείς συντεταγμένες
points1_h = [matched_points1, ones(size(matched_points1,1),1)];
points2_h = [matched_points2, ones(size(matched_points2,1),1)];

% Εκτέλεση RANSAC
for i = 1:ransac_max_iters
    % Τυχαία επιλογή 4 αντιστοιχιών
    rand_indices = randperm(size(matched_points1,1), 4);
    subset1 = matched_points1(rand_indices, :);
    subset2 = matched_points2(rand_indices, :);
    
    % Υπολογισμός ομοσχετίωσης (homography) με το υποσύνολο
    try
        H = fitgeotrans(subset1, subset2, 'projective');
        H_matrix = H.T;
    catch
        continue; % Παράκαμψη αν η ομοσχετίωση δεν μπορεί να υπολογιστεί
    end
    
    % Μετασχηματισμός όλων των σημείων της πρώτης εικόνας
    projected_points = (H_matrix * points1_h')';
    projected_points = projected_points(:,1:2) ./ projected_points(:,3);
    
    % Υπολογισμός αποστάσεων μεταξύ προβλεπόμενων και πραγματικών σημείων
    errors = sqrt(sum((projected_points - matched_points2).^2, 2));
    
    % Εύρεση inliers βάσει του ορίου αποστάσεων
    inliers = find(errors < ransac_threshold);
    
    % Ενημέρωση του καλύτερου μοντέλου αν βρεθούν περισσότερα inliers
    if length(inliers) > length(best_inliers)
        best_inliers = inliers;
        best_H = H_matrix;
    end
end

fprintf('Αριθμός inliers μετά το RANSAC: %d\n', length(best_inliers));

% Αποθήκευση των inlier αντιστοιχιών
if isempty(best_inliers)
    error('Δεν βρέθηκαν inliers μετά το RANSAC.');
end

inlier_points1 = matched_points1(best_inliers, :);
inlier_points2 = matched_points2(best_inliers, :);

%% Οπτικοποίηση Τελικών Αντιστοιχιών
fprintf('Οπτικοποίηση των τελικών αντιστοιχιών...\n');

% Φόρτωση των αρχικών εικόνων για εμφάνιση
img1_display = imread(image1_filename);
img2_display = imread(image2_filename);

% Μετατροπή σε RGB αν είναι grayscale
if size(img1_display,3) == 1
    img1_display = repmat(img1_display, [1,1,3]);
end
if size(img2_display,3) == 1
    img2_display = repmat(img2_display, [1,1,3]);
end

% Αλλαγή μεγέθους σε 256x256 αν χρειάζεται
img1_display = imresize(img1_display, [256, 256]);
img2_display = imresize(img2_display, [256, 256]);

% Δημιουργία συνενωμένης εικόνας
combined_img = [img1_display, img2_display];

% Προσαρμογή των συντεταγμένων της δεύτερης εικόνας
inlier_points2_shifted = inlier_points2;
inlier_points2_shifted(:,1) = inlier_points2_shifted(:,1) + size(img1_display,2);

% Δημιουργία νέας μορφής για εμφάνιση
figure('Color', 'white');
imshow(combined_img);
hold on;

% Σχεδίαση inlier σημείων με μικρότερο μέγεθος
plot(inlier_points1(:,1), inlier_points1(:,2), 'ro', 'MarkerSize', 3, 'LineWidth', 1);
plot(inlier_points2_shifted(:,1), inlier_points2_shifted(:,2), 'ro', 'MarkerSize', 3, 'LineWidth', 1);

% Δημιουργία χρωματικού χάρτη για τις γραμμές
num_matches = size(inlier_points1, 1);
colors = jet(ceil(num_matches/3)); % Χρησιμοποίηση του jet colormap για ποικιλία χρωμάτων
color_idx = randi(size(colors,1), num_matches, 1); % Τυχαία ανάθεση χρωμάτων

% Σχεδίαση γραμμών με διαφάνεια και διαφορετικά χρώματα
for i = 1:size(inlier_points1,1)
    line([inlier_points1(i,1), inlier_points2_shifted(i,1)], ...
         [inlier_points1(i,2), inlier_points2_shifted(i,2)], ...
         'Color', [colors(color_idx(i),:), 0.3], ... % Προσθήκη διαφάνειας 0.3
         'LineWidth', 0.5); % Λεπτότερες γραμμές
end

title('Αντιστοιχίσεις Σημείων-Κλειδιών με Inliers από RANSAC');
hold off;

% Προσθήκη legend
legend('Σημεία-κλειδιά', 'Location', 'southoutside');

fprintf('Η αντιστοίχιση ολοκληρώθηκε. Ελέγξτε το παράθυρο της εικόνας για τα αποτελέσματα.\n');