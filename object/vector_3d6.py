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
# Matplotlib's 3D plots don't have a built-in 'arrow' function,
# so we use this standard helper class.
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
FOCAL_LENGTH = 540 # Example value, replace with your calibrated one

# --- 2. KNOWN OBJECTS (Width in Centimeters) ---
# Event-focused list of common, small items
KNOWN_OBJECTS = {
    "cell phone": 7.6,  # Approx. width of a large smartphone
    "laptop": 35.0,     # Approx. width of a 15" laptop
    "bottle": 7.0,      # Water bottle
    "cup": 8.0,         # Coffee cup
    "book": 15.0,       # Avg. book width
    "backpack": 30.0,   # Standard backpack
    "handbag": 25.0,    # Handbag/purse
    "umbrella": 6.0,    # Closed umbrella
    "remote": 5.0,      # TV/projector remote
    "keyboard": 30.0,
    "mouse": 6.0,
}

# --- 3. DETECTION SETTINGS ---
CONFIDENCE_THRESHOLD = 0.5
MAX_VECTORS_TO_DRAW = 5 # Set to 5

# --- 4. 3D VISUALIZATION SETTINGS ---
INITIAL_VIEW_LIMIT = 100.0 # 100cm (1 meter)

# --- NEW: PLOT UPDATE SKIPS ---
# We still skip frames to reduce plot lag
PLOT_UPDATE_SKIP_FRAMES = 5 


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load YOLOv8 model
    model = YOLO("yolov8s.pt") # Using "small" (s) model for better accuracy
    model.to(device)
    print("Loaded YOLOv8s model.")

    # --- Setup 3D Plot ---
    plt.ion() # Turn on interactive mode
    fig_3d = plt.figure(figsize=(8, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    print("Initialized 3D plot window.")
    
    # --- Setup plot elements *outside* the loop ---
    ax_3d.set_title("3D Vector Visualization (Up to 5 Objects)")
    ax_3d.set_xlabel("X (cm)")
    ax_3d.set_zlabel("Y (cm)") # Y is up/down on screen
    ax_3d.set_ylabel("Z (cm)") # Z is depth
    
    # Plot camera origin
    ax_3d.plot([0], [0], [0], 'ko', markersize=10, label="Camera (Origin)")
    
    # --- NEW: Create a pool of 5 arrows and a list for text artists ---
    gaze_arrows = []
    vector_label_artists = [] # This list will hold the 3D text artists
    colors = plt.cm.jet(np.linspace(0, 1, MAX_VECTORS_TO_DRAW)) # Get 5 different colors

    for i in range(MAX_VECTORS_TO_DRAW):
        # Create arrow, make it invisible initially
        arrow = Arrow3D(
            [0, 0], [0, 0], [0, 0],
            mutation_scale=15, lw=3, arrowstyle="-|>", 
            color=colors[i], visible=False
        )
        ax_3d.add_artist(arrow)
        gaze_arrows.append(arrow)


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

    # --- Frame Counter ---
    prev_frame_time = time.time()
    fps_text = ""
    frame_count = 0 

    # --- Per-Frame Analysis Loop ---
    while True:
        success, frame = cap.read()
        if not success: 
            print("Failed to grab frame.")
            break
        
        frame_count += 1
        
        # --- Calculate FPS ---
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.1f}"
        
        # --- State for this frame ---
        detected_vectors = [] # NEW: List to store (X, Y, Z, name) tuples
        
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
                if name in KNOWN_OBJECTS and len(detected_vectors) < MAX_VECTORS_TO_DRAW: 
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
                        
                        # --- Add to our list for plotting ---
                        detected_vectors.append((X, Y, Z, name))
                        
                        # --- Draw on the live feed ---
                        label = f"{name} | Z: {Z:.2f}cm"
                        color_index = len(detected_vectors) - 1
                        # Convert matplotlib color (RGBA, 0-1) to OpenCV (BGR, 0-255)
                        cv_color = [int(c * 255) for c in colors[color_index][2::-1]] 
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), cv_color, 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv_color, 2)
            
        # --- Only update plot elements every N frames to reduce lag ---
        if frame_count % PLOT_UPDATE_SKIP_FRAMES == 0:
            
            # --- NEW: Clear all old 3D text labels ---
            for txt in vector_label_artists:
                txt.remove()
            vector_label_artists.clear()

            max_Z = INITIAL_VIEW_LIMIT # Default zoom

            for i in range(MAX_VECTORS_TO_DRAW):
                if i < len(detected_vectors):
                    # A vector was detected for this slot
                    X, Y, Z, name = detected_vectors[i]
                    
                    # Update arrow
                    arrow_data = ([0, X], [0, Z], [0, Y]) # (X, Z, Y)
                    gaze_arrows[i].set_data_3d(*arrow_data)
                    gaze_arrows[i].set_visible(True)
                    
                    # --- NEW: Create 3D text label at the tip of the arrow ---
                    label_text = f"{name}\n({X:.0f}, {Y:.0f}, {Z:.0f})cm"
                    # Plot text at the (X, Z, Y) coordinate (matching axes)
                    txt_artist = ax_3d.text(X, Z, Y, label_text, color=colors[i], fontsize=9)
                    vector_label_artists.append(txt_artist)
                    
                    # Update max zoom
                    if Z > max_Z:
                        max_Z = Z
                
                else:
                    # No vector for this slot, hide the arrow
                    gaze_arrows[i].set_visible(False)
            

            # --- Dynamically set plot limits to "zoom in" ---
            z_limit = max(100, max_Z + 100) # At least 100cm, or 100cm past farthest object
            ax_3d.set_ylim(0, z_limit) # Z is depth (plotted on Y-axis)
            
            xy_limit = max(50, z_limit * 0.5) # X/Y axes are half the Z axis
            ax_3d.set_xlim(-xy_limit, xy_limit)
            ax_3d.set_zlim(-xy_limit, xy_limit) # Y is height (plotted on Z-axis)
                    
            # --- Update the 3D plot ---
            # This is the line that causes lag, but it's necessary
            # in a single-threaded script to see updates.
            plt.pause(0.001)

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

