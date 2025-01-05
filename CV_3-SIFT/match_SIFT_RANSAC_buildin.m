% match_SIFT_RANSAC_buildin.m
% 
% Αυτό το σκριπτ υλοποιεί την αντιστοίχιση σημείων-κλειδιών μεταξύ δύο εικόνων
% χρησιμοποιώντας τον αλγόριθμο SIFT και RANSAC για την απομάκρυνση ακατάλληλων αντιστοιχίσεων.
%
% Προαπαιτούμενα:
% - Computer Vision Toolbox
%
% Χρήση:
% - Αντικαταστήστε τα 'image1.jpg' και 'image2.jpg' με τα ονόματα των εικόνων σας.
%
% Συγγραφέας: ΑΓΓΕΛΟΣ ΝΙΚΟΛΑΟΣ ΠΟΤΑΜΙΑΝΟΣ

%% Αρχικοποίηση
clear; clc; close all;

%% Ορισμός Ονομάτων Εικόνων
image1_filename = 'cameraman_zoom.png'; % Αντικαταστήστε με το όνομα της πρώτης εικόνας
image2_filename = 'cameraman_rotate.png'; % Αντικαταστήστε με το όνομα της δεύτερης εικόνας

%% Φόρτωση και Προεπεξεργασία Εικόνων
fprintf('Φόρτωση και προεπεξεργασία εικόνων...\n');

% Φόρτωση πρώτης εικόνας
img1 = imread(image1_filename);
if size(img1,3) == 3
    img1_gray = rgb2gray(img1);
else
    img1_gray = img1;
end
img1_gray = imresize(img1_gray, [256, 256]);
img1_gray = im2double(img1_gray);

% Φόρτωση δεύτερης εικόνας
img2 = imread(image2_filename);
if size(img2,3) == 3
    img2_gray = rgb2gray(img2);
else
    img2_gray = img2;
end
img2_gray = imresize(img2_gray, [256, 256]);
img2_gray = im2double(img2_gray);

%% Ανίχνευση Σημείων-Κλειδιών SIFT
fprintf('Ανίχνευση σημείων-κλειδιών SIFT...\n');
points1 = detectSIFTFeatures(img1_gray);
points2 = detectSIFTFeatures(img2_gray);

% Εξαγωγή Περιγραφέων
[features1, validPoints1] = extractFeatures(img1_gray, points1);
[features2, validPoints2] = extractFeatures(img2_gray, points2);

fprintf('Αριθμός σημείων-κλειδιών στην πρώτη εικόνα: %d\n', size(features1,1));
fprintf('Αριθμός σημείων-κλειδιών στη δεύτερη εικόνα: %d\n', size(features2,1));

%% Αντιστοίχιση Περιγραφέων
fprintf('Αντιστοίχιση περιγραφέων με χρήση Ratio Test...\n');
indexPairs = matchFeatures(features1, features2, 'MaxRatio', 0.8, 'Unique', true);

matchedPoints1 = validPoints1(indexPairs(:,1));
matchedPoints2 = validPoints2(indexPairs(:,2));

fprintf('Αριθμός αντιστοιχιών μετά το Ratio Test: %d\n', size(matchedPoints1,1));

%% Εκτίμηση Geometric Transform με RANSAC
fprintf('Εφαρμογή του αλγορίθμου RANSAC για εύρεση inliers...\n');
[tform, inlierIdx] = estimateGeometricTransform2D(matchedPoints1, matchedPoints2, ...
    'projective', 'MaxDistance', 3);

inlierPoints1 = matchedPoints1(inlierIdx, :);
inlierPoints2 = matchedPoints2(inlierIdx, :);

fprintf('Αριθμός inliers μετά το RANSAC: %d\n', size(inlierPoints1,1));

%% Οπτικοποίηση Τελικών Αντιστοιχιών
fprintf('Οπτικοποίηση των τελικών αντιστοιχιών...\n');

% Δημιουργία συνενωμένης εικόνας (πλευρική σύνδεση) με ασπρόμαυρες εικόνες
img1_display = imresize(img1_gray, [256, 256]);
img2_display = imresize(img2_gray, [256, 256]);

% Διασφάλιση ότι και οι δύο εικόνες έχουν το ίδιο πλήθος καναλιών
if size(img1_display, 3) == 1
    img1_display = cat(3, img1_display, img1_display, img1_display);
end

if size(img2_display, 3) == 1
    img2_display = cat(3, img2_display, img2_display, img2_display);
end

% Συνένωση των εικόνων
combined_img = [img1_display, img2_display];

% Προσαρμογή των συντεταγμένων της δεύτερης εικόνας
inlierPoints2_shifted = inlierPoints2;
inlierPoints2_shifted.Location(:,1) = inlierPoints2_shifted.Location(:,1) + size(img1_display,2);

% Δημιουργία νέας μορφής για εμφάνιση
figure;
imshow(combined_img);
hold on;

% Σχεδίαση inlier σημείων
plot(inlierPoints1.Location(:,1), inlierPoints1.Location(:,2), 'ro', 'MarkerSize',5, 'LineWidth',1.5);
plot(inlierPoints2_shifted.Location(:,1), inlierPoints2_shifted.Location(:,2), 'go', 'MarkerSize',5, 'LineWidth',1.5);

% Σχεδίαση γραμμών μεταξύ των inliers
for i = 1:size(inlierPoints1,1)
    line([inlierPoints1.Location(i,1), inlierPoints2_shifted.Location(i,1)], ...
         [inlierPoints1.Location(i,2), inlierPoints2_shifted.Location(i,2)], 'Color', 'y');
end

title('Αντιστοιχίσεις Σημείων-Κλειδιών με Inliers από RANSAC');
hold off;

fprintf('Η αντιστοίχιση ολοκληρώθηκε. Ελέγξτε το παράθυρο της εικόνας για τα αποτελέσματα.\n');
