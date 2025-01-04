# Computer Vision Lab

Welcome to the **Computer Vision Lab** repository! This repository contains exercises from the Computer Vision course at the University of Patras, focused on foundational and advanced techniques in computer vision. Each exercise builds on theoretical concepts and practical applications, utilizing tools like MATLAB and Python (OpenCV).

---

## Exercises Overview

### 1. Gaussian and Laplacian Pyramids ([Details](CV_PYRAMIDS.pdf))
- **Objective:** Image decomposition into multiple scales using Gaussian and Laplacian pyramids.
- **Topics Covered:**
  - Image denoising and enhancement.
  - Image blending and mosaicing techniques using Laplacian pyramids.
  - Multiscale representation theory.
- **Tools Used:** MATLAB (toolbox functions like `gen_Pyr`, `pyrBlend`).

### 2. Geometric Transformations ([Details](CV_TRANSFORMATIONS.pdf))
- **Objective:** Familiarize with geometric transformations like scaling, rotation, and shearing.
- **Topics Covered:**
  - Image manipulation and animation using affine transformations.
  - Video generation using custom transformations.
  - Interpolation methods and quality comparisons.
- **Tools Used:** MATLAB (`imread`, `imwrap`, `affine2d`, etc.)

### 3. Scale-Invariant Feature Transform (SIFT) ([Details](CV_3-SIFT.pdf))
- **Objective:** Understand and implement the SIFT algorithm for feature detection and description.
- **Topics Covered:**
  - Detection of keypoints and local extrema in scale-space.
  - Orientation assignment and keypoint descriptors.
  - Matching features between images for alignment.
- **Tools Used:** MATLAB/Python (OpenCV).

### 4. Image Alignment ([Details](CV_4-ALIGNMENT.pdf))
- **Objective:** Study and compare image alignment algorithms.
- **Topics Covered:**
  - Enhanced Correlation Coefficient (ECC) and Lucas-Kanade (LK) algorithms.
  - Robustness under photometric and geometric distortions.
  - Noise analysis and PSNR evaluation.
- **Tools Used:** MATLAB (ECC, LK alignment functions).

### 5. Autoencoders and Variational Autoencoders ([Details](CV_5_AUTOENCODERS.pdf))
- **Objective:** Explore dimensionality reduction and feature learning using Autoencoders (AEs) and Variational Autoencoders (VAEs).
- **Topics Covered:**
  - Principal Component Analysis (PCA) as a baseline.
  - Training AEs and VAEs on datasets like MNIST.
  - Reconstruction quality and visualization of learned features.
- **Tools Used:** Python (TensorFlow/Keras or PyTorch).

### 6. Convolutional Neural Networks (CNNs) for Object Detection ([Details](CV_CNN-1.pdf))
- **Objective:** Apply deep learning for object detection and localization.
- **Topics Covered:**
  - R-CNN architecture and its components (region proposal, feature extraction, classification).
  - Intersection over Union (IoU) metrics.
  - Training CNNs with datasets annotated for object detection.
- **Tools Used:** Python (TensorFlow/Keras or PyTorch).

---

## How to Use This Repository

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/nickpotamianos/Computer_Vision_Lab.git
   ```

2. **Setup Environment:**
   - Install dependencies for MATLAB or Python.
   - For Python-based exercises, set up a virtual environment with necessary libraries (e.g., OpenCV, TensorFlow).

3. **Navigate to Exercises:**
   - Each exercise has its dedicated folder with required scripts, data, and instructions.

4. **Follow the Instructions:**
   - Each folder contains a detailed PDF document for implementation guidance.


---

## Contributors

- [Nick Potamianos](https://github.com/nickpotamianos) / up1084537@ac.upatras.gr
- Teaching Staff at the University of Patras, Department of Computer Engineering and Informatics.

---

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.
