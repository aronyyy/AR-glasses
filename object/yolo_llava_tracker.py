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

# --- AI & ML IMPORTS ---
from ultralytics import YOLO
import torch
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from langchain_ollama import OllamaLLM

# ==========================================
#               CONFIGURATION
# ==========================================

# 1. VISION AI
OLLAMA_MODEL = "llava" 

# 2. DETECTOR
YOLO_MODEL = "yolo11s.pt" 

# 3. TRACKING
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4. TUNING
# Lowered to 0.15 to detect "almost anything"
CONF_THRESH = 0.1        
IMG_SZ = 640              
SIMILARITY_THRESHOLD = 0.55 
LAST_N_IDS = 10           
CONTEXT_PADDING = 50      

# 5. FILTERING
# COCO Class IDs to ignore: 0 = Person, 60 = Dining Table
IGNORED_CLASSES = {0, 60}

# ==========================================
#           SHARED MEMORY
# ==========================================
ollama_queue = queue.Queue()
ai_results = {}

current_scene_state = {
    "frame": None,
    "boxes": []
}

# ==========================================
#           1. THE AI WORKER
# ==========================================
def cv2_to_base64(cv2_img):
    img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.convert("RGB") 
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ai_worker_thread():
    print(f"[THREAD] Connecting to Ollama ({OLLAMA_MODEL})...")
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL)
        print("[THREAD] Warming up model...")
        llm.invoke("hello") 
        print("[THREAD] LLaVA is Ready!")
    except Exception as e:
        print(f"[THREAD ERROR] {e}")
        return

    while True:
        try:
            obj_id, crop_img = ollama_queue.get()
        except:
            continue

        if crop_img is None: 
            ollama_queue.task_done()
            continue

        try:
            prompt = "Describe this object in 5 words or less."
            img_b64 = cv2_to_base64(crop_img)
            llm_with_image = llm.bind(images=[img_b64])
            response = llm_with_image.invoke(prompt)
            clean_text = response.replace("\n", " ").strip()
            
            ai_results[obj_id] = clean_text
            print(f"[AI] ID {obj_id}: {clean_text}")
            
        except Exception as e:
            print(f"[THREAD ERROR] {e}")
        finally:
            ollama_queue.task_done()

# ==========================================
#           2. INTERACTION (MOUSE CLICK)
# ==========================================
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = current_scene_state["frame"]
        boxes = current_scene_state["boxes"]
        
        if frame is None or not boxes:
            return

        clicked_something = False
        for (x1, y1, x2, y2, obj_id) in boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"[CLICK] User clicked on ID {obj_id}")
                
                ai_results[obj_id] = "Thinking..."
                clicked_something = True
                
                h, w, _ = frame.shape
                p = CONTEXT_PADDING
                cx1, cy1 = max(0, x1-p), max(0, y1-p)
                cx2, cy2 = min(w, x2+p), min(h, y2+p)
                crop = frame[cy1:cy2, cx1:cx2].copy()
                
                ollama_queue.put((obj_id, crop))
                break
        
        if not clicked_something:
            print("[CLICK] Clicked empty space.")

# ==========================================
#           3. TRACKING HELPERS
# ==========================================
def make_embedder():
    print(f"[INFO] Loading ResNet50 on {DEVICE}...")
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(DEVICE)
    resnet.eval()
    extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
    transform = T.Compose([
        T.ToPILImage(), T.Resize((224,224)), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return extractor, transform

def get_embedding(extractor, transform, img):
    if img is None or img.size == 0: return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t_img = transform(rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = extractor(t_img).squeeze().cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ==========================================
#           4. MAIN LOOP
# ==========================================
def run_app(video_source):
    t = threading.Thread(target=ai_worker_thread, daemon=True)
    t.start()

    print(f"[INFO] Loading YOLO: {YOLO_MODEL}...")
    yolo = YOLO(YOLO_MODEL)
    embedder, transform = make_embedder()

    recent_ids = deque(maxlen=LAST_N_IDS) 
    id_embeddings = {}                    
    next_id = 0

    cap = cv2.VideoCapture(video_source if str(video_source) != "0" else 0)
    
    window_name = "Filter & Click (YOLO + LLaVA)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    print("\n[INFO] READY! Ignoring People & Tables.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- STEP 1: DETECTION (With Filter) ---
        results = yolo(frame, imgsz=IMG_SZ, conf=CONF_THRESH, verbose=False)
        
        detections = [] 
        crops = []
        
        for r in results:
            for b in r.boxes:
                # 1. Check Class ID
                cls_id = int(b.cls[0])
                
                # 2. Filter: If it's a Person (0) or Table (60), SKIP IT.
                if cls_id in IGNORED_CLASSES:
                    continue

                x1,y1,x2,y2 = map(int, b.xyxy[0])
                
                # Filter tiny noise
                if (x2-x1) < 20 or (y2-y1) < 20: continue
                
                detections.append((x1, y1, x2, y2))
                crops.append(frame[y1:y2, x1:x2])

        # --- STEP 2: TRACKING ---
        embeddings = [get_embedding(embedder, transform, c) for c in crops]
        current_frame_assignments = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det
            emb = embeddings[i]
            assigned_id = None
            
            if emb is not None and recent_ids:
                best_sim = 0
                best_match = None
                for rid in recent_ids:
                    if rid in id_embeddings:
                        sim = cosine_sim(emb, id_embeddings[rid])
                        if sim > best_sim:
                            best_sim = sim
                            best_match = rid
                if best_sim >= SIMILARITY_THRESHOLD:
                    assigned_id = best_match

            if assigned_id is None:
                assigned_id = next_id
                next_id += 1
                if emb is not None:
                    id_embeddings[assigned_id] = emb
                recent_ids.append(assigned_id)

            current_frame_assignments.append(assigned_id)

        # --- STEP 3: UPDATE MOUSE STATE ---
        current_frame_boxes = []
        for (det, tid) in zip(detections, current_frame_assignments):
             x1, y1, x2, y2 = det
             current_frame_boxes.append((x1, y1, x2, y2, tid))
        
        current_scene_state["frame"] = frame
        current_scene_state["boxes"] = current_frame_boxes

        # --- STEP 4: VISUALIZATION ---
        for (x1, y1, x2, y2, tid) in current_frame_boxes:
            text = ai_results.get(tid, "")
            
            if text == "Thinking...":
                color = (0, 165, 255) 
            elif text != "":
                color = (0, 255, 0)   
            else:
                color = (200, 200, 200) 

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            
            # Show ID
            label = f"ID {tid}"
            cv2.putText(frame, label, (x1, max(15, y1-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show Description
            if text and text != "Thinking...":
                cv2.putText(frame, text, (x1, y2 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif text == "Thinking...":
                cv2.putText(frame, "Analyzing...", (x1, y2 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="0")
    args = parser.parse_args()
    run_app(args.video)