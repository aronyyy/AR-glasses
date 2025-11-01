import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time # Import time for FPS calculation

# --- New 3D Plotting Imports ---
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

# --- 3D Arrow Class (from visual.py) ---
# ... (Arrow3D class code remains the same) ...
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        self._proj3d = None 

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self._proj3d = xs, ys, zs
        return zs[0]

    def draw(self, renderer):
        if self._proj3d is None:
            self.do_3d_projection(renderer)
            
        xs, ys, zs = self._proj3d
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
    def set_data_3d(self, xs, ys, zs):
        """Sets the new 3D coordinates for the arrow."""
        self._verts3d = xs, ys, zs
        self._proj3d = None # Force reprojection
# ---------------------------------------------------


# --- 1. CALIBRATION & INTRINSICS ---
# PASTE YOUR FOCAL LENGTH (in pixels) HERE
# This value is found by running calibrate_video.py
FOCAL_LENGTH = 540

# --- 2. KNOWN OBJECTS (Width in Centimeters) ---
# ... (KNOWN_OBJECTS dict remains the same) ...
KNOWN_OBJECTS = {
    "person": 45.0,     # Avg. shoulder width
    "book": 15.0,
    "cell phone": 7.6,  # Approx. width of a large smartphone
    "laptop": 35.0,     # Approx. width of a 15" laptop
    # Add your calibrated object name and width here if different
    "cup": 7.0, # Example
    "bottle": 7.0 # Example
}

# --- 3. DETECTION SETTINGS ---
CONFIDENCE_THRESHOLD = 0.5

# --- 4. 3D VISUALIZATION SETTINGS ---
INITIAL_VIEW_LIMIT = 100.0 # 100cm (1 meter)

# --- NEW: PLOT UPDATE SKIPS ---
# ... (PLOT_UPDATE_SKIP_FRAMES remains the same) ...
PLOT_UPDATE_SKIP_FRAMES = 5 


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load YOLOv8 model
    # --- CHANGED: Using the "small" (s) model for better accuracy ---
    model = YOLO("yolov8s.pt") 
    model.to(device)
    print("Loaded YOLOv8s model.")

    # --- Setup 3D Plot ---
    plt.ion() # Turn on interactive mode
    fig_3d = plt.figure(figsize=(8, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    print("Initialized 3D plot window.")
    
    # --- Setup plot elements *outside* the loop ---
    ax_3d.set_title("3D Vector Visualization")
    ax_3d.set_xlabel("X (cm)")
    ax_3d.set_zlabel("Y (cm)") # Y is up/down on screen
    ax_3d.set_ylabel("Z (cm)") # Z is depth
    
    # Plot camera origin
    ax_3d.plot([0], [0], [0], 'ko', markersize=10, label="Camera (Origin)")
    
    # Create the arrow artist once
    arrow_data = ([0, 0], [0, 0], [0, 0])
    gaze_arrow = Arrow3D(
        *arrow_data,
        mutation_scale=15, lw=3, arrowstyle="-|>", color='g'
    )
    ax_3d.add_artist(gaze_arrow)

    # Create the legend text object once
    vector_label = ax_3d.text2D(0.05, 0.95, "Searching...", transform=ax_3d.transAxes, color='g')

    # Set initial plot limits
    ax_3d.set_ylim(0, INITIAL_VIEW_LIMIT) # Z (depth)
    ax_3d.set_xlim(-INITIAL_VIEW_LIMIT/2, INITIAL_VIEW_LIMIT/2) # X
    ax_3d.set_zlim(-INITIAL_VIEW_LIMIT/2, INITIAL_VIEW_LIMIT/2) # Y

    # --- Webcam Setup ---
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print(f"Error: Could not open webcam.")
        plt.ioff()
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Estimate camera intrinsics
    camera_intrinsics = {
        "fx": FOCAL_LENGTH, "fy": FOCAL_LENGTH,
        "cx": float(frame_width / 2), "cy": float(frame_height / 2)
    }
    print(f"Using {frame_width}x{frame_height} video with F_x={FOCAL_LENGTH:.2f}")

    # --- Store last known vector ---
    last_X, last_Y, last_Z = 0, 0, 0
    
    # --- NEW: Frame Counter ---
    prev_frame_time = time.time()
    fps_text = ""
    frame_count = 0 # <-- NEW

    # --- Per-Frame Analysis Loop ---
    while True:
        success, frame = cap.read()
        if not success: 
            print("Failed to grab frame.")
            break
        
        # --- NEW: Increment frame counter ---
        frame_count += 1
        
        # --- NEW: Calculate FPS ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.1f}"
        
        # --- State for this frame ---
        object_detected_this_frame = False
        label_text = "Searching..."
        label_color = 'r'
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        
        # Process detections
        for r in results:
            for box in r.boxes:
                # Get class name
                cls = int(box.cls[0])
                name = model.names[cls]
                confidence = float(box.conf)

                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                
                # --- Check if this is one of our known objects ---
                if name == "bottle": 
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Get object info from our dict
                    actual_width = KNOWN_OBJECTS[name]
                    pixel_width = x2 - x1
                    
                    if pixel_width > 0:
                        # Z = Distance in cm
                        Z = (actual_width * FOCAL_LENGTH) / pixel_width

                        # X = Horizontal coordinate in cm
                        X = (center_x - camera_intrinsics["cx"]) * Z / camera_intrinsics["fx"]
                        
                        # Y = Vertical coordinate in cm (negative is up)
                        Y = (center_y - camera_intrinsics["cy"]) * Z / camera_intrinsics["fy"]
                        
                        # --- Update state for this frame ---
                        last_X, last_Y, last_Z = X, Y, Z
                        object_detected_this_frame = True
                        label_text = f"Vector: ({X:.1f}, {Y:.1f}, {Z:.1f}) cm"
                        label_color = 'g'

                        # Print the 3D vector to the console
                        print(f"Object: {name}, Vector: (X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}) cm")
                        
                        dist_text = f"Z: {Z:.2f}cm"
                        
                        # --- Draw on the live feed ---
                        label = f"{name} | {dist_text}"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Only detect one object for clarity
                        break 
            
            # --- Update the vector ---
            if not object_detected_this_frame and (last_X, last_Y, last_Z) != (0, 0, 0):
                # Object was detected before, but not this frame. Show last known position.
                label_text = f"Last: ({last_X:.1f}, {last_Y:.1f}, {last_Z:.1f}) cm"
                label_color = 'r' # Red for "last known"
            
            # --- NEW: Only update plot elements every N frames ---
            if frame_count % PLOT_UPDATE_SKIP_FRAMES == 0:
                # Update the arrow's 3D data
                arrow_data = ([0, last_X], [0, last_Z], [0, last_Y]) # (X, Z, Y)
                gaze_arrow.set_data_3d(*arrow_data)
                gaze_arrow.set_color(label_color)

                # Update the vector label text
                vector_label.set_text(label_text)
                vector_label.set_color(label_color)

                # --- Dynamically set plot limits to "zoom in" ---
                z_limit = max(100, last_Z + 100) # At least 100cm, or 100cm past object
                ax_3d.set_ylim(0, z_limit) # Z is depth (plotted on Y-axis)
                
                xy_limit = max(50, z_limit * 0.5) # X/Y axes are half the Z axis
                ax_3d.set_xlim(-xy_limit, xy_limit)
                ax_3d.set_zlim(-xy_limit, xy_limit) # Y is height (plotted on Z-axis)
                
                # --- Update the 3D plot ---
                fig_3d.canvas.draw_idle()
                fig_3d.canvas.flush_events()

        # --- Add FPS counter to CV2 window ---
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # --- Show the live annotated frame ---
        cv2.imshow("Realtime 3D Detection", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    print("Stream ended.")

if __name__ == '__main__':
    main()

