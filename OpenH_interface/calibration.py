import subprocess
from datetime import datetime
import tkinter as tk
import time

# ==========================
# Experiment metadata
# ==========================
PROJECT_NAME = "eye_calib"

FPS = 25
DURATION = 60000  # ms

CAM0_RES = (640, 480)
CAM1_RES = (1920, 1080)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

cam0_filename = (
    f"{PROJECT_NAME}_cam0_{CAM0_RES[0]}x{CAM0_RES[1]}_"
    f"{FPS}fps_{ts}.mp4"
)

cam1_filename = (
    f"{PROJECT_NAME}_cam1_{CAM1_RES[0]}x{CAM1_RES[1]}_"
    f"{FPS}fps_{ts}.mp4"
)

# ==========================
# Camera recording commands
# ==========================
cmd0 = [
    "rpicam-vid",
    "--camera", "0",
    "--width", str(CAM0_RES[0]),
    "--height", str(CAM0_RES[1]),
    "--framerate", str(FPS),
    "-t", str(DURATION),
    "--codec", "h264",
    "-o", cam0_filename
]

cmd1 = [
    "rpicam-vid",
    "--camera", "1",
    "--width", str(CAM1_RES[0]),
    "--height", str(CAM1_RES[1]),
    "--framerate", str(FPS),
    "-t", str(DURATION),
    "--codec", "h264",
    "-o", cam1_filename
]

# Start recording
p0 = subprocess.Popen(cmd0)
p1 = subprocess.Popen(cmd1)

# ==========================
# Calibration window
# ==========================
POINT_INTERVAL = 3
POINT_RADIUS = 12
MARGIN_RATIO = 0.1

root = tk.Tk()
root.overrideredirect(True)
root.attributes("-fullscreen", True)
root.geometry(
    f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0"
)
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

rows = 4
cols = 5

x_margin = int(screen_w * MARGIN_RATIO)
y_margin = int(screen_h * MARGIN_RATIO)

x_positions = [
    x_margin + i * (screen_w - 2 * x_margin) // (cols - 1)
    for i in range(cols)
]

y_positions = [
    y_margin + i * (screen_h - 2 * y_margin) // (rows - 1)
    for i in range(rows)
]

points = [(x, y) for y in y_positions for x in x_positions]

for x, y in points:
    canvas.delete("all")
    canvas.create_oval(
        x - POINT_RADIUS,
        y - POINT_RADIUS,
        x + POINT_RADIUS,
        y + POINT_RADIUS,
        fill="white",
        outline="white"
    )
    root.update()
    time.sleep(POINT_INTERVAL)

root.destroy()

# ==========================
# Wait for recording to end
# ==========================
p0.wait()
p1.wait()

print("Recording and calibration complete")
print("Saved files:")
print(cam0_filename)
print(cam1_filename)