%% partA_ex5.m
%
% Άσκηση 5: Έλεγχος ρωμαλεότητας (ECC & LK) στην παρουσία φωτονικών παραμορφώσεων
%
% - Υποθέτουμε ότι στο ecc_lk.m έχουν σχολιαστεί οι γραμμές κανονικοποίησης (template=..., image=...),
%   ώστε οι φωτομετρικές αλλοιώσεις να μη μηδενίζονται μέσα στη συνάρτηση.
% - Δοκιμάζουμε διαφορετικές τιμές contrast & brightness
% - Ελέγχουμε 3 περιπτώσεις: παραμόρφωση ΜΟΝΟ στο template, ΜΟΝΟ στο image, ΚΑΙ στα δύο
% - Αποθηκεύουμε τα MSE (ECC), MSE (LK) και rho (ECC).
% - Τέλος, κάνουμε έναν συγκεντρωτικό πίνακα/γράφημα για να διαπιστώσουμε πόσο «ανθεκτικός» είναι ο κάθε αλγόριθμος.

clear; clc; close all;

%% 1) Επιλογή δύο frames αρκετά μακριά
videoFile = 'video1_high.avi';  % ή 'video2_high.avi'
if ~isfile(videoFile)
    error('Το βίντεο δεν βρέθηκε στον τρέχοντα φάκελο.');
end

vObj = VideoReader(videoFile);

frameIndexTemplate = 1;
frameIndexImage    = 40;  % αρκετά πιο «μακριά»

if frameIndexImage > vObj.NumFrames
    error('Το βίντεο έχει λιγότερα από %d frames!', frameIndexImage);
end

frame1 = read(vObj, frameIndexTemplate);
frame2 = read(vObj, frameIndexImage);

% Μετατροπή σε grayscale αν είναι έγχρωμα
if size(frame1,3) == 3
    frame1 = rgb2gray(frame1);
end
if size(frame2,3) == 3
    frame2 = rgb2gray(frame2);
end

% Μετατροπή σε double χωρίς επιπλέον κανονικοποίηση (κρατάμε όπως είναι):
template = double(frame1);
image    = double(frame2);

%% 2) Παράμετροι ECC/LK
num_levels   = 1;          
num_iterations = 15;       
transform    = 'affine';  
init_warp    = eye(2,3);   % Identity initialization

%% 3) Λίστες contrast & brightness
% Μπορείτε να αλλάξετε αυτές τις τιμές ελεύθερα.
contrastList   = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0];
brightnessList = [-50, -30, 0, 30, 60];

% Θα αποθηκεύουμε τα αποτελέσματα σε struct (ή πίνακα).
ResultsStruct = struct();
idxCase = 0;

%% 4) Βρόχος δοκιμών
for c = 1:length(contrastList)
    for b = 1:length(brightnessList)
        
        aVal = contrastList(c);
        bVal = brightnessList(b);
        
        % -------- Περίπτωση (A) Παραμόρφωση ΜΟΝΟ στο template --------
        idxCase = idxCase + 1;
        disp('=============================================================');
        disp(['Case ', num2str(idxCase), ...
              ' (Παραμόρφωση ΜΟΝΟ στο template) | Contrast=', num2str(aVal), ...
              ' | Brightness=', num2str(bVal)]);

        template_mod = aVal * template + bVal;  
        image_mod    = image;  % αμετάβλητο

        [resA, resA_lk, MSE_A, rho_A, MSELK_A] = ecc_lk( ...
            image_mod, ...
            template_mod, ...
            num_levels, ...
            num_iterations, ...
            transform, ...
            init_warp );

        ResultsStruct(idxCase).type = 'Template Modified Only';
        ResultsStruct(idxCase).contrast   = aVal;
        ResultsStruct(idxCase).brightness = bVal;
        ResultsStruct(idxCase).MSE_ECC    = MSE_A(end);
        ResultsStruct(idxCase).MSE_LK     = MSELK_A(end);
        ResultsStruct(idxCase).rho_ECC    = rho_A(end);

        % -------- Περίπτωση (B) Παραμόρφωση ΜΟΝΟ στο image --------
        idxCase = idxCase + 1;
        disp('-------------------------------------------------------------');
        disp(['Case ', num2str(idxCase), ...
              ' (Παραμόρφωση ΜΟΝΟ στο image) | Contrast=', num2str(aVal), ...
              ' | Brightness=', num2str(bVal)]);

        template_mod = template;
        image_mod    = aVal * image + bVal;

        [resB, resB_lk, MSE_B, rho_B, MSELK_B] = ecc_lk( ...
            image_mod, ...
            template_mod, ...
            num_levels, ...
            num_iterations, ...
            transform, ...
            init_warp );

        ResultsStruct(idxCase).type = 'Image Modified Only';
        ResultsStruct(idxCase).contrast   = aVal;
        ResultsStruct(idxCase).brightness = bVal;
        ResultsStruct(idxCase).MSE_ECC    = MSE_B(end);
        ResultsStruct(idxCase).MSE_LK     = MSELK_B(end);
        ResultsStruct(idxCase).rho_ECC    = rho_B(end);

        % -------- Περίπτωση (C) Παραμόρφωση ΚΑΙ στο template & image --------
        idxCase = idxCase + 1;
        disp('-------------------------------------------------------------');
        disp(['Case ', num2str(idxCase), ...
              ' (Παραμόρφωση ΚΑΙ σε template & image) | Contrast=', ...
               num2str(aVal), ' | Brightness=', num2str(bVal)]);

        template_mod = aVal * template + bVal;
        image_mod    = aVal * image    + bVal;

        [resC, resC_lk, MSE_C, rho_C, MSELK_C] = ecc_lk( ...
            image_mod, ...
            template_mod, ...
            num_levels, ...
            num_iterations, ...
            transform, ...
            init_warp );

        ResultsStruct(idxCase).type = 'Both Modified';
        ResultsStruct(idxCase).contrast   = aVal;
        ResultsStruct(idxCase).brightness = bVal;
        ResultsStruct(idxCase).MSE_ECC    = MSE_C(end);
        ResultsStruct(idxCase).MSE_LK     = MSELK_C(end);
        ResultsStruct(idxCase).rho_ECC    = rho_C(end);
        
        disp('=============================================================');
        disp(' ');
    end
end

%% 5) Παρουσίαση Συγκεντρωτικών Αποτελεσμάτων
disp('====== Συνοπτικά Αποτελέσματα για Όλες τις Περιπτώσεις ======');

nCases = length(ResultsStruct);
for i = 1:nCases
    disp(['Case #', num2str(i), ...
          ' | Type: ', ResultsStruct(i).type, ...
          ' | Contrast=', num2str(ResultsStruct(i).contrast), ...
          ' | Brightness=', num2str(ResultsStruct(i).brightness), ...
          ' | MSE(ECC)=', num2str(ResultsStruct(i).MSE_ECC), ...
          ' | MSE(LK)=',  num2str(ResultsStruct(i).MSE_LK), ...
          ' | rho(ECC)=', num2str(ResultsStruct(i).rho_ECC)]);
end

% Μετατρέπουμε τα πεδία σε πίνακες για πιο εύκολη απεικόνιση
allMSE_ECC = [ResultsStruct(:).MSE_ECC];
allMSE_LK  = [ResultsStruct(:).MSE_LK];
allRho_ECC = [ResultsStruct(:).rho_ECC];

% Φτιάχνουμε ένα διάγραμμα με 3 υπο-διαγράμματα
figure('Name','Άσκηση 5: Σύγκριση ECC & LK υπό διαφορετικά contrast/brightness');
subplot(3,1,1);
stem(allMSE_ECC, 'b','filled');
title('Τελικό MSE (ECC) σε όλες τις περιπτώσεις');
ylabel('MSE'); grid on;

subplot(3,1,2);
stem(allMSE_LK, 'r','filled');
title('Τελικό MSE (LK) σε όλες τις περιπτώσεις');
ylabel('MSE'); grid on;

subplot(3,1,3);
stem(allRho_ECC, 'g','filled');
title('\rho (ECC) σε όλες τις περιπτώσεις');
xlabel('Αριθμός Περίπτωσης'); ylabel('\rho'); grid on;

%% 6) Απλός «δείκτης αποτυχίας» (προαιρετικό)
% π.χ. ορίζουμε μια τιμή-κατώφλι για MSE ή/και για rho
% και βλέπουμε σε πόσες περιπτώσεις "κόκκινες γραμμές" περνάει ο καθένας

% Ενδεικτικά thresholds:
MSE_fail_threshold = 40;   % αν το MSE > 40
rho_fail_threshold = 0.85; % αν η ρ < 0.85
failCount_ECC = 0;
failCount_LK  = 0;

for i = 1:nCases
    if ResultsStruct(i).MSE_ECC > MSE_fail_threshold || ...
       ResultsStruct(i).rho_ECC < rho_fail_threshold
        failCount_ECC = failCount_ECC + 1;
    end
    if ResultsStruct(i).MSE_LK > MSE_fail_threshold
        failCount_LK = failCount_LK + 1;
    end
end

fprintf('\n=== Πρόχειρος δείκτης «αποτυχίας» ===\n');
fprintf('ECC fail count: %d / %d\n', failCount_ECC, nCases);
fprintf('LK  fail count: %d / %d\n', failCount_LK,  nCases);

