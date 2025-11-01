import cv2
import torch
from ultralytics import YOLO

# --- 1. USER CONFIGURATION ---

# IMPORTANT: Paste the FOCAL_LENGTH value you get from running calibrate_video.py
FOCAL_LENGTH = 540  # <<< PASTE YOUR CALIBRATED VALUE HERE

# Define the known REAL-WORLD widths (in meters) of the objects you want to track
KNOWN_OBJECTS = {
    "bottle": 0.07,      # 7 cm (Your calibration object)
    "book": 0.15,        # 15 cm
    "laptop": 0.35,      # 35 cm
    "person": 0.45,      # 45 cm (average shoulder width)
    "cell phone": 0.075, # 7.5 cm
    "cup": 0.08          # 8 cm
    # Add any other objects from the COCO dataset
}

# --- END CONFIGURATION ---

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the fast YOLOv8n model
    model = YOLO('yolov8n.pt')
    model.to(device)
    print("Loaded YOLOv8n model.")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get camera properties for intrinsics
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate camera intrinsic parameters (principal point)
    # We assume the principal point is the center of the image
    cx = frame_width / 2
    cy = frame_height / 2
    
    # We use our calibrated FOCAL_LENGTH for both fx and fy
    # This assumes square pixels
    fx = FOCAL_LENGTH
    fy = FOCAL_LENGTH

    print(f"Camera Intrinsics (est.): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # --- Realtime Loop ---
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break
        
        # Run YOLO detection
        results = model(frame, verbose=False)

        # Process detections
        for r in results:
            for box in r.boxes:
                # Get class name
                cls = int(box.cls[0])
                name = model.names[cls]

                # Check if it's an object we know the size of
                if name in KNOWN_OBJECTS:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Calculate pixel width and center
                    pixel_width = x2 - x1
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    if pixel_width > 0:
                        # Get actual object width
                        actual_width = KNOWN_OBJECTS[name]

                        # --- 3D VECTOR CALCULATION ---
                        
                        # 1. Calculate Z (Distance)
                        Z = (actual_width * FOCAL_LENGTH) / pixel_width

                        # 2. Calculate X
                        # X = (pixel_x - principal_point_x) * Z / focal_length_x
                        X = (center_x - cx) * Z / fx

                        # 3. Calculate Y
                        # Y = (pixel_y - principal_point_y) * Z / focal_length_y
                        Y = (center_y - cy) * Z / fy
                        
                        # --- End Calculation ---

                        # Print the 3D vector to the console
                        print(f"Object: {name:<12} | Vector (X,Y,Z): ({X:+.2f}m, {Y:+.2f}m, {Z:.2f}m)")

                        # Draw on the frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Prepare text for display
                        vec_text = f"Z: {Z:.2f}m" # Show Z (distance)
                        vec_text_xyz = f"X:{X:+.1f}, Y:{Y:+.1f}, Z:{Z:.1f}"
                        
                        cv2.putText(frame, f"{name}", (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, vec_text, (int(x1), int(y1) - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Realtime 3D Detection (YOLOv8 + Known Width)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream ended.")

if __name__ == '__main__':
    main()

