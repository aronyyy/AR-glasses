import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

from vector_3d_output import process_frame  # your gaze tracker


def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def main():
    # 1. Select files
    print("Please select the calibration matrix file...")
    matrix_path = select_file(
        "Select calibration_matrix.npy",
        [("Numpy files", "*.npy")]
    )
    eye_vid_path = select_file(
        "Select Eye Video",
        [("Video files", "*.mp4 *.avi *.mkv *.mov *.flv")]
    )
    world_vid_path = select_file(
        "Select World Video",
        [("Video files", "*.mp4 *.avi *.mkv *.mov *.flv")]
    )

    if not all([matrix_path, eye_vid_path, world_vid_path]):
        print("Missing file selection, exiting.")
        return

    # 2. Load homography and videos
    H_mat = np.load(matrix_path)

    cap_e = cv2.VideoCapture(eye_vid_path)
    cap_w = cv2.VideoCapture(world_vid_path)

    fps = cap_w.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback
    frames_per_point = int(3 * fps)  # kept for alignment if needed

    print("Overlaying gaze onto world video...")

    while True:
        ret_e, f_eye = cap_e.read()
        ret_w, f_world = cap_w.read()
        if not ret_e or not ret_w:
            break

        # 4. Get gaze direction from your tracker
        ellipse, sphere_center, direction = process_frame(f_eye)

        if direction is not None:
            dx = float(direction[0])
            dy = float(direction[1])

            # Map (dx, dy) through homography to screen/world coords
            eye_point = np.array([dx, dy, 1.0], dtype=np.float32)
            res = H_mat @ eye_point

            if abs(res[2]) > 1e-6:
                gaze_x = int(res[0] / res[2])
                gaze_y = int(res[1] / res[2])

                # Draw gaze dot (red)
                cv2.circle(f_world, (gaze_x, gaze_y), 20, (0, 0, 255), -1)
                cv2.circle(f_world, (gaze_x, gaze_y), 23, (255, 255, 255), 2)

        # Show result
        disp = cv2.resize(f_world, (960, 540))
        cv2.imshow("Calibration Validation Overlay", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_e.release()
    cap_w.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
