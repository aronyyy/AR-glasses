import cv2
import numpy as np
from ultralytics import SAM

# -----------------------------
# Load SAM model
# -----------------------------
# If you want speed -> use "mobile_sam.pt" or "fast_sam.pt"
sam = SAM("mobile_sam.pt")  
# sam = SAM("sam_b.pt")     # better quality but slower

def mask_to_box(mask):
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()

# -----------------------------
# Webcam Stream
# -----------------------------
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run SAM segmentation (AUTO mode)
    results = sam(frame)

    # Convert to CPU numpy
    res = results[0].cpu()

    # Extract masks
    boxes = []
    masks_obj = getattr(res, "masks", None)

    if masks_obj is not None and hasattr(masks_obj, "data"):
        mask_tensor = masks_obj.data  # (N,H,W)

        for i in range(mask_tensor.shape[0]):
            mask_np = mask_tensor[i].cpu().numpy() > 0.5

            # skip very tiny objects
            if mask_np.sum() < 200:
                continue

            b = mask_to_box(mask_np)
            if b is None:
                continue

            boxes.append(b)

    # Draw boxes
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("SAM Real-Time Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
