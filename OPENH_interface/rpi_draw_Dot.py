import cv2
import numpy as np
from vector_3d_output import process_frame  # your gaze tracker


CALIB_FILE = "calibration_matrix.npy"
EYE_CAM_INDEX = 0      # Pi eye camera
WORLD_CAM_INDEX = 1    # Pi front/screen camera


def main():
    # 1. Load calibration matrix
    try:
        H_mat = np.load(CALIB_FILE)
    except Exception as e:
        print("Could not load calibration matrix:", e)
        return

    # 2. Open eye and world cameras
    cap_e = cv2.VideoCapture(EYE_CAM_INDEX)
    cap_w = cv2.VideoCapture(WORLD_CAM_INDEX)

    if not cap_e.isOpened():
        print("Error: could not open eye camera.")
        return
    if not cap_w.isOpened():
        print("Error: could not open world camera.")
        return

    win_name = "Pi Gaze Overlay"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("Press 'q' to quit.")

    while True:
        ret_e, f_eye = cap_e.read()
        ret_w, f_world = cap_w.read()
        if not ret_e or not ret_w:
            break

        # If your vector_3d_output expects flips/resizes, do them here.
        # Example (only if needed):
        # f_eye_proc = cv2.flip(f_eye, -1)
        f_eye_proc = f_eye

        # 3. Get gaze direction from your tracker
        ellipse, sphere_center, direction = process_frame(f_eye_proc)

        if direction is not None:
            dx = float(direction[0])
            dy = float(direction[1])

            # 4. Map (dx, dy) through homography to screen/world coords
            eye_point = np.array([dx, dy, 1.0], dtype=np.float32)
            res = H_mat @ eye_point

            if abs(res[2]) > 1e-6:
                gaze_x = int(res[0] / res[2])
                gaze_y = int(res[1] / res[2])

                h, w = f_world.shape[:2]
                gaze_x = max(0, min(w - 1, gaze_x))
                gaze_y = max(0, min(h - 1, gaze_y))

                # 5. Draw gaze dot (red) on the world frame
                cv2.circle(f_world, (gaze_x, gaze_y), 15, (0, 0, 255), -1)
                cv2.circle(f_world, (gaze_x, gaze_y), 18, (255, 255, 255), 2)

        cv2.imshow(win_name, f_world)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_e.release()
    cap_w.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
