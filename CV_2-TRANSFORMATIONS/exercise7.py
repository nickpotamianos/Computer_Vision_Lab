import cv2
import numpy as np


def create_beach_ball_animation():
    # Read images
    ball = cv2.imread('ball.jpg', cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('ball_mask.jpg', cv2.IMREAD_GRAYSCALE)
    background = cv2.imread('beach.jpg')

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Resize background to a standard size
    background = cv2.resize(background, (1280, 720))

    # Resize ball and mask to be proportional to background
    ball_size = 80
    ball = cv2.resize(ball, (ball_size, ball_size))
    mask = cv2.resize(mask, (ball_size, ball_size))

    # Get dimensions
    height, width = background.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('transf_beach_ex7.avi', fourcc, 30.0, (width, height))

    # Animation parameters
    num_frames = 120

    # Bouncing parameters
    gravity = 0.5
    bounce_height = 200
    horizontal_speed = 8

    # Starting position
    x = width // 4
    y = height - ball_size - 50  # Start near bottom of frame

    # Initial vertical velocity (negative means going up)
    velocity = -20

    # Rotation angle
    angle = 0

    for frame in range(num_frames):
        # Update position
        x += horizontal_speed
        velocity += gravity
        y += velocity

        # Bounce when hitting the ground
        if y > height - ball_size - 50:  # Ground position
            y = height - ball_size - 50
            velocity = -abs(velocity * 0.7)  # Reduce bounce height by 30%

        # Update rotation
        angle += 5  # Rotate 5 degrees per frame

        # Create rotation matrix
        M = cv2.getRotationMatrix2D((ball_size // 2, ball_size // 2), angle, 1.0)

        # Rotate ball and mask
        rotated_ball = cv2.warpAffine(ball, M, (ball_size, ball_size))
        rotated_mask = cv2.warpAffine(mask, M, (ball_size, ball_size))

        # Create frame
        frame = background.copy()

        # Calculate ROI coordinates
        y1 = int(y - ball_size // 2)
        y2 = y1 + ball_size
        x1 = int(x - ball_size // 2)
        x2 = x1 + ball_size

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

        # Reset ball position if it goes off screen
        if x > width - ball_size:
            x = ball_size

    # Release video writer
    out.release()


try:
    create_beach_ball_animation()
    print("Beach ball animation created successfully!")
except Exception as e:
    print(f"An error occurred: {str(e)}")