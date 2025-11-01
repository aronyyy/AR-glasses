import cv2
import torch
import numpy as np
from ultralytics import YOLO

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
        self._proj3d = None # Will store projected 2D coords

    def do_3d_projection(self, renderer=None):
        """
        Perform the 3D projection for this artist.
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self._proj3d = xs, ys, zs
        # Return the z-value of the arrow's base for sorting
        return zs[0]

    def draw(self, renderer):
        """
        Draw the artist.
        """
        if self._proj3d is None:
            self.do_3d_projection(renderer)
            
        xs, ys, zs = self._proj3d
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
# ---------------------------------------------------


# --- 1. CALIBRATION & INTRINSICS ---
# <<< PASTE Your FOCAL_LENGTH from calibrate_headless.py HERE
FOCAL_LENGTH = 750.0 # Example: 785.22

# --- 2. KNOWN OBJECTS (Width in Centimeters) ---
# <<< ADD your new objects here
KNOWN_OBJECTS = {
    # Add the object you just calibrated (e.g., "cup": 7.0)
    "YOUR_CALIBRATED_OBJECT_NAME": 7.0, 
    "rubix cube": 5.0, # Your new object (5cm)
    "bottle": 7.0,     # Standard bottle (7cm)
    "cell phone": 7.6, # Approx. width of a large phone (7.6cm)
    "book": 15.0,
    "person": 45.0,     # Approx. shoulder width
    "laptop": 35.0,     # <<< ADDED LAPTOP (Approx 35cm width)
}

# --- 3. DETECTION SETTINGS ---
CONFIDENCE_THRESHOLD = 0.5

# --- 4. 3D VISUALIZATION SETTINGS ---
# MAX_Z_CM = 500 # No longer needed, will autoscale

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the fast YOLOv8n model
    model = YOLO('yolov8n.pt')
    model.to(device)
    print("Loaded YOLOv8n model.")

    # --- Setup 3D Plot ---
    plt.ion() # Turn on interactive mode
    fig_3d = plt.figure(figsize=(8, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    print("Initialized 3D plot window.")
    
    # --- Webcam Setup ---
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print(f"Error: Could not open webcam.")
        return

    # Get webcam properties to calculate intrinsics
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- DYNAMIC INTRINSICS CALCULATION ---
    # We use our calibrated FOCAL_LENGTH
    camera_intrinsics = {
        "fx": FOCAL_LENGTH, "fy": FOCAL_LENGTH,
        "cx": float(frame_width / 2), "cy": float(frame_height / 2)
    }
    print(f"Using {frame_width}x{frame_height} video with F_x={FOCAL_LENGTH:.2f}")

    # --- Store last known vector ---
    last_X, last_Y, last_Z = 0, 0, 0

    # --- Per-Frame Analysis Loop ---
    while True:
        success, frame = cap.read()
        if not success: 
            print("Failed to grab frame.")
            break
        
        # --- Clear 3D plot for new frame ---
        ax_3d.cla()

        # --- State for this frame ---
        object_detected_this_frame = False
        vector_label = "No target detected"

        # --- Setup 3D plot axes ---
        ax_3d.set_title("3D Vector Visualization")
        # We will plot (X, Z, Y) to make Z the "forward" axis
        ax_3d.set_xlabel("X (cm)")
        ax_3d.set_zlabel("Y (cm)") # Y is up/down on screen
        ax_3d.set_ylabel("Z (cm)") # Z is depth
        
        # Plot camera origin
        ax_3d.plot([0], [0], [0], 'ko', markersize=10, label="Camera (Origin)")

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
                if name == "laptop": # <<< CHANGED: Only detect "laptop"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    pixel_width = x2 - x1
                    actual_width = KNOWN_OBJECTS[name] # Width is now in cm

                    # --- THIS IS THE 3D CALCULATION ---
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
                        vector_label = f"Vector: ({X:.1f} cm, {Y:.1f} cm, {Z:.1f} cm)"

                        # Print the 3D vector to the console
                        print(f"Object: {name}, Vector: (X={X:.2f} cm, Y={Y:.2f} cm, Z={Z:.2f} cm)")
                        
                        dist_text = f"Z: {Z:.2f}cm"
                        
                        # --- Draw on the live feed ---
                        label = f"{name} | {dist_text}"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Only detect one object for clarity
                        break 
            
            # --- Draw the vector ---
            if not object_detected_this_frame and (last_X, last_Y, last_Z) != (0, 0, 0):
                # Object was detected before, but not this frame. Show last known position.
                vector_label = f"Last: ({last_X:.1f} cm, {last_Y:.1f} cm, {last_Z:.1f} cm)"
                color = 'r' # Red for "last known"
            else:
                color = 'g' # Green for "live" or "none"
            
            # Draw the arrow (if there's a vector to draw)
            if (last_X, last_Y, last_Z) != (0, 0, 0):
                gaze_arrow = Arrow3D(
                    [0, last_X], [0, last_Z], [0, last_Y], # Plotting (X, Z, Y)
                    mutation_scale=15, lw=3, arrowstyle="-|>", color=color
                )
                ax_3d.add_artist(gaze_arrow)

            # --- Dynamically set plot limits to "zoom in" ---
            # Set Z (depth) limit from 0 to slightly more than the object's distance
            z_limit = max(100, last_Z + 100) # At least 100cm, or 100cm past object
            ax_3d.set_ylim(0, z_limit) # Z is depth (plotted on Y-axis)
            
            # Set X/Y limits to be proportional to the Z limit to keep a nice view
            xy_limit = max(50, z_limit * 0.5) # X/Y axes are half the Z axis
            ax_3d.set_xlim(-xy_limit, xy_limit)
            ax_3d.set_zlim(-xy_limit, xy_limit) # Y is height (plotted on Z-axis)

            # Update plot legend
            ax_3d.plot([], [], [], f'{color}-', lw=3, label=vector_label) # Dummy plot for legend
            ax_3d.legend(loc='upper left')

            # --- Show the live annotated frame ---
            # This will fail in a headless notebook, but is needed for local VS Code
            try:
                cv2.imshow("Realtime 3D Detection", annotated_frame)
            except Exception as e:
                if 'The function is not implemented' in str(e):
                    print("GUI not supported. Printing to console only.")
                    # Break here if we're in a headless environment to avoid flooding
                    if not cap.isOpened(): 
                        break
                else:
                    print(f"cv2.imshow error: {e}")
                    break
        
        # --- Update the 3D plot window ---
        plt.pause(0.001)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff() # Turn off interactive mode
    plt.show() # Keep the final plot window open
    print("Stream ended.")

if __name__ == '__main__':
    main()


