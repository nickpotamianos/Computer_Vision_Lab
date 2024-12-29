import cv2
import numpy as np

def create_windmill_animation():
    # Read images
    windmill = cv2.imread('windmill.png', cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('windmill_mask.png', cv2.IMREAD_GRAYSCALE)
    background = cv2.imread('windmill_back.jpeg')

    # Invert the mask so the blades are white and background is black
    mask = cv2.bitwise_not(mask)

    # Find the bounding box of the windmill blades to crop
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # Crop all images to the bounding box
    mask = mask[y:y + h, x:x + w]
    windmill = windmill[y:y + h, x:x + w]

    # Define smaller dimensions (about 1/2 of the original size to match reference)
    new_width = mask.shape[1] // 2
    new_height = mask.shape[0] // 2

    # Resize all images
    mask = cv2.resize(mask, (new_width, new_height))
    windmill = cv2.resize(windmill, (new_width, new_height))
    background = cv2.resize(background, (background.shape[1] // 2, background.shape[0] // 2))  # Keep background larger

    # Calculate position to place the blades in the center of background
    y_offset = (background.shape[0] - new_height) // 2
    x_offset = (background.shape[1] - new_width) // 2

    # Get dimensions for video
    height, width = background.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('transf_windmill.avi', fourcc, 30.0, (width, height))

    # Animation parameters
    num_frames = 200  # Increased number of frames
    angle_step = -360 / num_frames  # Negative for clockwise rotation, smaller step for slower rotation

    # Get center of rotation for the blades
    center = (new_width // 2, new_height // 2)

    for frame in range(num_frames):
        # Calculate rotation angle
        angle = frame * angle_step

        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate windmill and mask
        rotated_windmill = cv2.warpAffine(windmill, M, (new_width, new_height))
        rotated_mask = cv2.warpAffine(mask, M, (new_width, new_height))

        # Create frame by combining background and rotated windmill
        frame = background.copy()

        # Create ROI in the frame
        roi = frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width]

        # Extract RGB channels from rotated windmill
        windmill_rgb = rotated_windmill[:, :, :3]

        # Create binary mask
        binary_mask = rotated_mask > 0

        # Apply mask to ROI
        for c in range(3):
            roi[:, :, c] = np.where(binary_mask, windmill_rgb[:, :, c], roi[:, :, c])

        # Put ROI back into frame
        frame[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = roi

        # Write frame
        out.write(frame)

    # Release video writer
    out.release()

try:
    create_windmill_animation()
    print("Animation created successfully!")
except Exception as e:
    print(f"An error occurred: {str(e)}")