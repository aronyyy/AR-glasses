import cv2
from tkinter import Tk, filedialog
from vector_11 import process_frame  # now returns (ellipse, sphere_center_3d)
import numpy as np


OUTPUT_FILE = "eye_calib_sphere_center_from_video.txt"

def pick_video_file():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select recorded EYE video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv")]
    )
    root.destroy()
    return video_path

def compute_eyeball_center_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: could not open video:", video_path)
        return None

    centers = []  # will hold 3D sphere_center vectors
    frame_count = 0

    while True:
       ret, frame = cap.read()
       if not ret:
         break

       frame_count += 1

       ellipse, sphere_center = process_frame(frame)

    # sphere_center should be a 3D vector (x, y, z)
       if sphere_center is not None:
         sc = np.asarray(sphere_center, dtype=float)
         if sc.shape == (3,):           # only accept proper 3D centers
             centers.append(sc)

       if frame_count % 100 == 0:
         print(f"Processed {frame_count} frames, collected {len(centers)} centers")

       if cv2.waitKey(1) & 0xFF == ord('q'):
         break


    cap.release()
    cv2.destroyAllWindows()

    if not centers:
        print("No valid sphere centers collected.")
        return None

    # Convert to float and average component-wise

    centers_np = np.stack(centers, axis=0)   # shape (N, 3)
    mean_center = centers_np.mean(axis=0)    # (3,)
    final_center = (float(mean_center[0]), float(mean_center[1]), float(mean_center[2]))

    with open(OUTPUT_FILE, "w") as f:
        f.write(f"{final_center[0]:.6f},{final_center[1]:.6f},{final_center[2]:.6f}\n")

    print("Final averaged sphere center (3D):", final_center)
    print("Saved to:", OUTPUT_FILE)
    return final_center

if __name__ == "__main__":
    path = pick_video_file()
    if not path:
        print("No video selected, exiting.")
    else:
        compute_eyeball_center_from_video(path)
