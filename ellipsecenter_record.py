import cv2
import time
from datetime import datetime
import tkinter as tk

# vector.process_frame must return: (final_rotated_rect, model_center_average)
from vector_11 import process_frame

# ==========================
# Experiment metadata
# ==========================

PROJECT_NAME = "eye_calib"
FPS = 25
DURATION_SEC = 60  # 1 minute

# Camera indices (adjust if needed)
EYE_CAM_INDEX = 0      # eye-facing camera
FRONT_CAM_INDEX = 1    # world/front camera

CAM0_RES = (640, 480)    # eye cam resolution
CAM1_RES = (1920, 1080)  # front cam resolution

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

eye_filename = (
    f"{PROJECT_NAME}_eye_{CAM0_RES[0]}x{CAM0_RES[1]}_"
    f"{FPS}fps_{ts}.mp4"
)

front_filename = (
    f"{PROJECT_NAME}_front_{CAM1_RES[0]}x{CAM1_RES[1]}_"
    f"{FPS}fps_{ts}.mp4"
)

EYEBALL_CENTER_FILE = f"{PROJECT_NAME}_eyeball_center_{ts}.txt"

# ==========================
# Calibration window helpers
# ==========================

POINT_INTERVAL = 3   # seconds per point
POINT_RADIUS = 12
MARGIN_RATIO = 0.1
ROWS = 4
COLS = 5

def setup_point_window():
    """Create full-screen black window with canvas and generate 4x5 grid points."""
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-fullscreen", True)
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
    root.configure(bg="black")
    root.focus_force()

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    canvas = tk.Canvas(
        root,
        width=screen_w,
        height=screen_h,
        bg="black",
        highlightthickness=0
    )
    canvas.pack(fill="both", expand=True)

    x_margin = int(screen_w * MARGIN_RATIO)
    y_margin = int(screen_h * MARGIN_RATIO)

    x_positions = [
        x_margin + i * (screen_w - 2 * x_margin) // (COLS - 1)
        for i in range(COLS)
    ]
    y_positions = [
        y_margin + i * (screen_h - 2 * y_margin) // (ROWS - 1)
        for i in range(ROWS)
    ]

    points = [(x, y) for y in y_positions for x in x_positions]
    return root, canvas, points

def draw_point(canvas, x, y):
    """Draw a single white calibration dot at (x, y)."""
    canvas.delete("all")
    canvas.create_oval(
        x - POINT_RADIUS,
        y - POINT_RADIUS,
        x + POINT_RADIUS,
        y + POINT_RADIUS,
        fill="white",
        outline="white",
    )

# ==========================
# Main calibration + recording
# ==========================

def run_calibration():
    # Open cameras
    eye_cap = cv2.VideoCapture(EYE_CAM_INDEX)
    front_cap = cv2.VideoCapture(FRONT_CAM_INDEX)

    eye_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM0_RES[0])
    eye_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM0_RES[1])
    eye_cap.set(cv2.CAP_PROP_FPS, FPS)

    front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM1_RES[0])
    front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM1_RES[1])
    front_cap.set(cv2.CAP_PROP_FPS, FPS)

    if not eye_cap.isOpened() or not front_cap.isOpened():
        print("Error: could not open one or both cameras")
        return None, None, None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    eye_writer = cv2.VideoWriter(eye_filename, fourcc, FPS, CAM0_RES)
    front_writer = cv2.VideoWriter(front_filename, fourcc, FPS, CAM1_RES)

    # Store per-frame eyeball centers (model_center_average from vector.py)
    eyeball_centers = []

    # Create calibration UI
    root, canvas, points = setup_point_window()

    start_time = time.time()
    next_point_change = start_time
    point_index = 0

    # Record and collect eyeball centers for DURATION_SEC
    while time.time() - start_time < DURATION_SEC:
        now = time.time()

        # Update calibration point every POINT_INTERVAL seconds
        if point_index < len(points) and now >= next_point_change:
            x, y = points[point_index]
            draw_point(canvas, x, y)
            root.update()
            point_index += 1
            next_point_change = now + POINT_INTERVAL

        ret_eye, eye_frame = eye_cap.read()
        ret_front, front_frame = front_cap.read()
        if not ret_eye or not ret_front:
            break

        # Process eye frame with vector.py
        # vector.process_frame must return:
        #   final_rotated_rect, model_center_average
        ellipse, eyeball_center = process_frame(eye_frame)

        # Collect eyeball center (model_center_average) for minute-long averaging
        if eyeball_center is not None:
            eyeball_centers.append(eyeball_center)
            # Optional visualization on the recorded eye video
            cv2.circle(eye_frame, eyeball_center, 4, (0, 255, 0), -1)

        # Write videos
        eye_writer.write(eye_frame)
        front_writer.write(front_frame)

        # Keep Tk responsive
        root.update_idletasks()

    # Close calibration window
    root.destroy()

    # Done with recording
    eye_cap.release()
    front_cap.release()
    eye_writer.release()
    front_writer.release()

    # Compute averaged eyeball center over the full minute
    if eyeball_centers:
        avg_x = int(sum(p[0] for p in eyeball_centers) / len(eyeball_centers))
        avg_y = int(sum(p[1] for p in eyeball_centers) / len(eyeball_centers))
        global_eyeball_center = (avg_x, avg_y)
    else:
        global_eyeball_center = None

    # Write final eyeball center to file
    if global_eyeball_center is not None:
        with open(EYEBALL_CENTER_FILE, "w") as f:
            f.write(f"{global_eyeball_center[0]},{global_eyeball_center[1]}\n")
        print("Eyeball center saved to:", EYEBALL_CENTER_FILE)
    else:
        print("No valid eyeball center estimated; file not written.")

    print("Recording and calibration complete")
    print("Saved files:")
    print("Eye cam:", eye_filename)
    print("Front cam:", front_filename)
    print("Average eyeball center:", global_eyeball_center)

    return global_eyeball_center, eye_filename, front_filename

if __name__ == "__main__":
    run_calibration()
