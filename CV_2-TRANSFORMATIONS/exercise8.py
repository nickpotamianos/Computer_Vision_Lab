import cv2
import numpy as np


def create_beach_ball_animation():
    # Read images
    ball = cv2.imread('ball.jpg', cv2.IMREAD_UNCHANGED)
    # Read and invert the mask
    mask = cv2.imread('ball_mask.jpg', cv2.IMREAD_GRAYSCALE)
    mask = cv2.bitwise_not(mask)  # Invert the mask so ball is white and background is black
    background = cv2.imread('beach.jpg')

    # Resize background to a standard size
    background = cv2.resize(background, (1280, 720))

    # Resize ball and mask to be proportional to background
    ball_size = 100
    ball = cv2.resize(ball, (ball_size, ball_size))
    mask = cv2.resize(mask, (ball_size, ball_size))

    # Get dimensions
    height, width = background.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('transf_beach.avi', fourcc, 30.0, (width, height))

    # Animation parameters
    num_frames = 150

    # Ball movement parameters
    start_x = width // 4
    start_y = height // 2

    # Define ball path (arc motion)
    t = np.linspace(0, 1, num_frames)
    x_positions = np.linspace(start_x, width * 3 // 4, num_frames)
    y_positions = start_y - 200 * np.sin(np.pi * t)  # Create arc motion

    # Scale factors (ball gets smaller as it moves)
    scale_factors = np.linspace(1.0, 0.3, num_frames)

    # Rotation angles
    rotation_angles = np.linspace(0, 360, num_frames)

    for frame in range(num_frames):
        # Get current position and scale
        current_x = int(x_positions[frame])
        current_y = int(y_positions[frame])
        scale = scale_factors[frame]
        angle = rotation_angles[frame]

        # Scale ball and mask
        current_size = int(ball_size * scale)
        if current_size < 1:  # Prevent size from becoming too small
            current_size = 1
        scaled_ball = cv2.resize(ball, (current_size, current_size))
        scaled_mask = cv2.resize(mask, (current_size, current_size))

        # Create rotation matrix
        M = cv2.getRotationMatrix2D((current_size // 2, current_size // 2), angle, 1.0)

        # Rotate ball and mask
        rotated_ball = cv2.warpAffine(scaled_ball, M, (current_size, current_size))
        rotated_mask = cv2.warpAffine(scaled_mask, M, (current_size, current_size))

        # Create frame
        frame = background.copy()

        # Calculate ROI coordinates
        y1 = current_y - current_size // 2
        y2 = y1 + current_size
        x1 = current_x - current_size // 2
        x2 = x1 + current_size

        # Ensure coordinates are within frame bounds
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > width: x2 = width
        if y2 > height: y2 = height

        # Adjust ball and mask size if needed
        ball_roi = rotated_ball[:(y2 - y1), :(x2 - x1)]
        mask_roi = rotated_mask[:(y2 - y1), :(x2 - x1)]

        # Create ROI in the frame
        roi = frame[y1:y2, x1:x2]

        # Apply mask
        mask_roi = mask_roi > 128  # Create binary mask
        roi[mask_roi] = ball_roi[mask_roi]

        # Put ROI back into frame
        frame[y1:y2, x1:x2] = roi

        # Write frame
        out.write(frame)

    # Release video writer
    out.release()


try:
    create_beach_ball_animation()
    print("Beach ball animation created successfully!")
except Exception as e:
    print(f"An error occurred: {str(e)}")