import argparse
import time
import cv2
import numpy as np
import threading
import queue
import base64
from io import BytesIO
from collections import deque
from PIL import Image

# AI / ML Imports
from ultralytics import YOLO
import torch
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from langchain_ollama import OllamaLLM

# -------- CONFIG ----------
YOLO_MODEL = "yolo11n.pt"  # or yolov8n.pt
CONF_THRESH = 0.35
IMG_SZ = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tracking settings
SIMILARITY_THRESHOLD = 0.6
LAST_N_IDS = 10
MIN_BOX_SIDE = 12
IGNORED_CLASSES = {"person", "dining table", "table"}

# Ollama Settings
OLLAMA_MODEL_NAME = "llava"
CONTEXT_PADDING = 50  # Pixels to add around the box for context

# --------------------------

# --- GLOBAL SHARED STATE ---
# Queue to send images to the AI worker
ollama_queue = queue.Queue()
# Dictionary to store results: { object_id: "Description string" }
ai_descriptions = {}

# --------------------------
#     OLLAMA WORKER
# --------------------------
def cv2_to_base64(cv2_img):
    """Convert CV2 BGR image to Base64 string for LLaVA"""
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ollama_worker():
    """Background thread that processes images one by one"""
    print("[THREAD] Loading LLaVA model in background...")
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL_NAME)
        # Pre-warm the model
        llm.invoke("hello")
        print("[THREAD] LLaVA Model Ready!")
    except Exception as e:
        print(f"[THREAD ERROR] Could not load Ollama: {e}")
        return

    while True:
        # Wait for a task from the main video loop
        # task structure: (object_id, cropped_image)
        try:
            obj_id, crop_img = ollama_queue.get()
        except:
            continue

        if crop_img is None: continue

        try:
            # Prepare prompt
            prompt = "Describe this object in 10 words or less."
            
            # Convert image
            img_b64 = cv2_to_base64(crop_img)
            
            # Run Inference
            llm_with_image = llm.bind(images=[img_b64])
            response = llm_with_image.invoke(prompt)
            
            # Clean up response (remove newlines)
            clean_response = response.replace("\n", " ").strip()
            
            # Store result globally so main thread can see it
            ai_descriptions[obj_id] = clean_response
            print(f"[{OLLAMA_MODEL_NAME}] ID {obj_id}: {clean_response}")
            
        except Exception as e:
            print(f"[THREAD ERROR] Inference failed: {e}")
        finally:
            ollama_queue.task_done()

# --------------------------
#     TRACKING HELPERS
# --------------------------
def make_resnet50_embedder(device):
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    resnet.eval()
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feature_extractor.eval()
    transform = T.Compose([
        T.ToPILImage(), T.Resize((224,224)), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return feature_extractor, transform, 2048

def get_embedding_resnet(feature_extractor, transform, crop_bgr, device):
    if crop_bgr is None or crop_bgr.size == 0: return None
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = transform(crop_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = feature_extractor(x).squeeze().cpu().numpy()
    return vec / (np.linalg.norm(vec) + 1e-8)

def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if b.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (b_norm @ a_norm)

# --------------------------
#     MAIN LOOP
# --------------------------
def run(video_source):
    # 1. Start AI Thread
    t = threading.Thread(target=ollama_worker, daemon=True)
    t.start()

    # 2. Load Models
    try:
        yolo = YOLO(YOLO_MODEL)
    except:
        yolo = YOLO("yolov8n.pt")
        
    feat_extractor, transform, feat_dim = make_resnet50_embedder(DEVICE)

    # 3. State
    recent_ids = deque(maxlen=LAST_N_IDS)
    id_to_embedding = {}
    next_id = 0

    cap = cv2.VideoCapture(video_source if str(video_source) != "0" else 0)
    
    print("[INFO] Starting video. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Detection
        results = yolo(frame, imgsz=IMG_SZ, conf=CONF_THRESH, verbose=False)
        detections = [] 
        crops = []
        
        # Parse YOLO
        for r in results:
            for b in r.boxes:
                conf = float(b.conf[0])
                cls_id = int(b.cls[0])
                cls_name = yolo.names[cls_id]
                
                if cls_name in IGNORED_CLASSES: continue

                x1,y1,x2,y2 = map(int, b.xyxy[0])
                if (x2-x1) < MIN_BOX_SIDE or (y2-y1) < MIN_BOX_SIDE: continue
                
                detections.append((x1, y1, x2, y2, cls_name))
                crops.append(frame[y1:y2, x1:x2])

        # Embeddings
        embeddings = []
        for crop in crops:
            embeddings.append(get_embedding_resnet(feat_extractor, transform, crop, DEVICE))

        assigned_ids_this_frame = []

        # Tracking Logic
        for i, det in enumerate(detections):
            x1, y1, x2, y2, cls_name = det
            emb = embeddings[i]
            assigned = None
            
            # Try to match existing ID
            if emb is not None and len(recent_ids) > 0:
                valid_refs = [rid for rid in recent_ids if rid in id_to_embedding]
                if valid_refs:
                    ref_embs = np.vstack([id_to_embedding[rid] for rid in valid_refs])
                    sims = cosine_sim(emb, ref_embs)
                    best_idx = np.argmax(sims)
                    if sims[best_idx] >= SIMILARITY_THRESHOLD:
                        assigned = valid_refs[best_idx]

            # --- NEW OBJECT DETECTED ---
            if assigned is None:
                assigned = next_id
                next_id += 1
                if emb is not None:
                    id_to_embedding[assigned] = emb
                recent_ids.append(assigned)
                
                # --- TRIGGER OLLAMA HERE ---
                # 1. Add context padding
                h, w, _ = frame.shape
                pad = CONTEXT_PADDING
                cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
                
                # 2. Crop with context
                context_crop = frame[cy1:cy2, cx1:cx2].copy()
                
                # 3. Send to background thread
                # We put it in ai_descriptions as "Analyzing..." immediately
                ai_descriptions[assigned] = "Analyzing..."
                ollama_queue.put((assigned, context_crop))
                # ---------------------------

            assigned_ids_this_frame.append(assigned)

        # Drawing
        for (det, tid) in zip(detections, assigned_ids_this_frame):
            x1, y1, x2, y2, cls_name = det
            
            # Determine color
            color = (0, 255, 0) # Green for known
            
            # Fetch AI Description
            desc = ai_descriptions.get(tid, "")
            if desc == "Analyzing...":
                color = (0, 165, 255) # Orange for analyzing
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            
            # Label: ID + Class
            label = f"ID:{tid} {cls_name}"
            cv2.putText(frame, label, (x1, max(15, y1-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Label: AI Description (Draw below box)
            if desc:
                cv2.putText(frame, desc, (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("YOLO + ResNet + LLaVA", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="0", help="Video path or 0")
    args = parser.parse_args()
    run(args.video)