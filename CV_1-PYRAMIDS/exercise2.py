import cv2
import numpy as np

def generate_gaussian_pyramid(img, levels):
    """Generate Gaussian pyramid for an image."""
    gaussian = img.copy()
    pyramids = [gaussian]
    
    for i in range(levels):
        # Use a larger Gaussian kernel for smoother downsampling
        gaussian = cv2.GaussianBlur(gaussian, (5, 5), 0)
        gaussian = cv2.pyrDown(gaussian)
        pyramids.append(gaussian)
    
    return pyramids

def generate_laplacian_pyramid(gaussian_pyramids):
    """Generate Laplacian pyramid from Gaussian pyramid."""
    laplacian_pyramids = []
    
    for i in range(len(gaussian_pyramids)-1):
        size = (gaussian_pyramids[i].shape[1], gaussian_pyramids[i].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyramids[i+1], dstsize=size)
        # Apply Gaussian blur to reduce artifacts
        gaussian_expanded = cv2.GaussianBlur(gaussian_expanded, (5, 5), 0)
        laplacian = cv2.subtract(gaussian_pyramids[i], gaussian_expanded)
        laplacian_pyramids.append(laplacian)
    
    laplacian_pyramids.append(gaussian_pyramids[-1])
    return laplacian_pyramids

def blend_pyramids(laplacian1, laplacian2, mask_pyramids):
    """Blend two Laplacian pyramids using Gaussian mask pyramids."""
    blended_pyramids = []
    
    for l1, l2, mask in zip(laplacian1, laplacian2, mask_pyramids):
        # Normalize mask to range [0, 1]
        mask_norm = mask / 255.0 if mask.dtype == np.uint8 else mask
        # Apply weighted blending
        blended = l1 * mask_norm + l2 * (1.0 - mask_norm)
        blended_pyramids.append(blended)
    
    return blended_pyramids

def reconstruct_image(laplacian_pyramids):
    """Reconstruct image from Laplacian pyramid."""
    reconstructed = laplacian_pyramids[-1]
    
    for i in range(len(laplacian_pyramids)-2, -1, -1):
        size = (laplacian_pyramids[i].shape[1], laplacian_pyramids[i].shape[0])
        reconstructed = cv2.pyrUp(reconstructed, dstsize=size)
        reconstructed = cv2.add(reconstructed, laplacian_pyramids[i])
    
    return reconstructed

def create_refined_mask(shape, eye_region):
    """Create a refined mask focusing on the eye region with smooth transitions."""
    mask = np.zeros(shape, dtype=np.float32)
    
    # Create main elliptical mask for eye region
    cv2.ellipse(mask, 
                center=eye_region['center'],
                axes=eye_region['axes'],
                angle=eye_region['angle'],
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=-1)
    
    # Apply multiple Gaussian blurs with different kernel sizes for smooth transition
    mask_blurred = cv2.GaussianBlur(mask, (31, 31), 10)
    mask_blurred = cv2.GaussianBlur(mask_blurred, (21, 21), 8)
    
    return mask_blurred

def main():
    # Read images
    woman = cv2.imread('photos/woman.png', 0)  # Read as grayscale
    hand = cv2.imread('photos/hand.png', 0)    # Read as grayscale
    
    # Ensure consistent image sizes
    target_size = (400, 400)  # Adjust based on your needs
    woman = cv2.resize(woman, target_size)
    hand = cv2.resize(hand, target_size)
    
    # Define eye region parameters (adjust these values based on your images)
    eye_region = {
        'center': (200, 180),  # Adjust these coordinates
        'axes': (70, 50),      # Adjust the size of the ellipse
        'angle': -15           # Adjust the rotation angle
    }
    
    # Create refined mask
    mask = create_refined_mask(woman.shape, eye_region)
    
    # Number of pyramid levels (increased for better detail preservation)
    levels = 6

    # Generate Gaussian pyramids
    gaussian_woman = generate_gaussian_pyramid(woman, levels)
    gaussian_hand = generate_gaussian_pyramid(hand, levels)
    gaussian_mask = generate_gaussian_pyramid(mask, levels)

    # Generate Laplacian pyramids
    laplacian_woman = generate_laplacian_pyramid(gaussian_woman)
    laplacian_hand = generate_laplacian_pyramid(gaussian_hand)

    # Blend pyramids
    blended_pyramids = blend_pyramids(laplacian_woman, laplacian_hand, gaussian_mask)

    # Reconstruct final image
    final = reconstruct_image(blended_pyramids)

    # Enhance contrast of the final image
    final = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]]) / 9
    final = cv2.filter2D(final, -1, kernel)

    # Normalize and convert to uint8
    final = np.clip(final, 0, 255).astype(np.uint8)

    # Save result
    cv2.imwrite('blended_result_refined.png', final)

if __name__ == "__main__":
    main()