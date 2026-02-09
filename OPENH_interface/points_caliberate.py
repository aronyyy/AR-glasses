import cv2
import numpy as np
from vector_3d_output import process_frame  # or from vector_11 import process_frame

def get_calib_matrix_averaged(eye_vid_path):
    # 1. GENERATE THE 20 SCREEN POINTS (4x5 Grid)
    SCREEN_W, SCREEN_H = 1920, 1080
    ROWS, COLS = 4, 5
    MARGIN = 0.1
    x_margin = int(SCREEN_W * MARGIN)
    y_margin = int(SCREEN_H * MARGIN)
    x_pos = [x_margin + i * (SCREEN_W - 2 * x_margin) // (COLS - 1) for i in range(COLS)]
    y_pos = [y_margin + i * (SCREEN_H - 2 * y_margin) // (ROWS - 1) for i in range(ROWS)]
    world_points = [(x, y) for y in y_pos for x in x_pos]

    # 2. INITIALIZE VIDEO
    cap = cv2.VideoCapture(eye_vid_path)
    fps = 25              # your capture FPS
    frames_per_point = 75 # 3 seconds @ 25fps

    eye_data_averages = []
    final_world_points = []

    print(f"Starting averaging process for {len(world_points)} points...")

    for i in range(len(world_points)):
        block_dx = []
        block_dy = []

        # Skip first 15 frames for this point (0.6s)
        start_f = i * frames_per_point + 15
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        # Use the next 60 frames (~2.4s)
        for _ in range(60):
            ret, frame = cap.read()
            if not ret:
                break

            # directly get gaze direction from your tracker
            ellipse, sphere_center, direction = process_frame(frame)

            if direction is not None:
                dx = float(direction[0])
                dy = float(direction[1])
                block_dx.append(dx)
                block_dy.append(dy)

        if len(block_dx) > 10:
            avg_x = float(np.mean(block_dx))
            avg_y = float(np.mean(block_dy))
            eye_data_averages.append([avg_x, avg_y])
            final_world_points.append(world_points[i])
            print(f"Point {i+1} Averaged: ({avg_x:.4f}, {avg_y:.4f}) using {len(block_dx)} frames")
        else:
            print(f"Point {i+1} Failed: Not enough gaze data captured")

    cap.release()

    # 3. COMPUTE HOMOGRAPHY
    if len(eye_data_averages) >= 4:
        src_pts = np.array(eye_data_averages, dtype=np.float32)   # (N, 2) in gaze-direction space
        dst_pts = np.array(final_world_points, dtype=np.float32)  # (N, 2) screen coords

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        np.save("calibration_matrix.npy", H)
        print("\nSUCCESS: Matrix saved as calibration_matrix.npy")
        return H
    else:
        print("\nERROR: Not enough valid points found.")
        return None

# Execute (change path to your eye video used for point calibration)
get_calib_matrix_averaged(r"C:\Users\aarus\OneDrive\Documents\ARGlasses\OPENH\eye_test1.mp4")
