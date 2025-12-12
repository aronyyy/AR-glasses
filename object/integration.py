import argparse
import time
import cv2
import numpy as np
import threading
import queue
import base64
import random
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. TUNING
CONF_THRESH = 0.15        
IMG_SZ = 640              
SIMILARITY_THRESHOLD = 0.55 
LAST_N_IDS = 10            
CONTEXT_PADDING = 50       

# 4. GAZE SENSITIVITY
# Since we are using a video file, the mapping won't be "real" (your real eye isn't moving),
# but this allows you to see the red dot move based on the video.
GAZE_SENSITIVITY_X = 1500
GAZE_SENSITIVITY_Y = 1500
DWELL_THRESHOLD = 15 

# 5. FILTERING
IGNORED_CLASSES = {0, 60} 

# ==========================================
#            SHARED MEMORY
# ==========================================
ollama_queue = queue.Queue()
ai_results = {}
gaze_dwell_counter = {}

# ==========================================
#          PART 1: THE EYE PROCESSOR
# ==========================================
class EyeGazeProcessor:
    def __init__(self):
        self.ray_lines = [] 
        self.model_centers = []
        self.max_rays = 100
        self.prev_model_center_avg = (320, 240)
        self.max_observed_distance = 202 
        self.stored_intersections = []

    def crop_to_aspect_ratio(self, image, width=640, height=480):
        if image is None: return None
        current_height, current_width = image.shape[:2]
        desired_ratio = width / height
        current_ratio = current_width / current_height

        if current_ratio > desired_ratio:
            new_width = int(desired_ratio * current_height)
            offset = (current_width - new_width) // 2
            cropped_img = image[:, offset:offset+new_width]
        else:
            new_height = int(current_width / desired_ratio)
            offset = (current_height - new_height) // 2
            cropped_img = image[offset:offset+new_height, :]
        return cv2.resize(cropped_img, (width, height))

    def get_darkest_area(self, image):
        ignoreBounds = 20
        imageSkipSize = 10
        searchArea = 20
        internalSkipSize = 5
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_sum = float('inf')
        darkest_point = None

        for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
            for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
                current_sum = np.int64(0)
                num_pixels = 0
                for dy in range(0, searchArea, internalSkipSize):
                    if y + dy >= gray.shape[0]: break
                    for dx in range(0, searchArea, internalSkipSize):
                        if x + dx >= gray.shape[1]: break
                        current_sum += gray[y + dy][x + dx]
                        num_pixels += 1
                if current_sum < min_sum and num_pixels > 0:
                    min_sum = current_sum
                    darkest_point = (x + searchArea // 2, y + searchArea // 2)
        return darkest_point

    def apply_binary_threshold(self, image, darkestPixelValue, addedThreshold):
        threshold = darkestPixelValue + addedThreshold
        _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        return thresholded_image

    def mask_outside_square(self, image, center, size):
        x, y = center
        half_size = size // 2
        mask = np.zeros_like(image)
        top_left_x = max(0, x - half_size)
        top_left_y = max(0, y - half_size)
        bottom_right_x = min(image.shape[1], x + half_size)
        bottom_right_y = min(image.shape[0], y + half_size)
        mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
        return cv2.bitwise_and(image, mask)

    def filter_contours(self, contours, pixel_thresh, ratio_thresh):
        max_area = 0
        largest_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= pixel_thresh:
                x, y, w, h = cv2.boundingRect(contour)
                current_ratio = max(max(w, h) / min(w, h), 0)
                if current_ratio <= ratio_thresh:
                    if area > max_area:
                        max_area = area
                        largest_contour = contour
        return [largest_contour] if largest_contour is not None else []

    def compute_gaze_vector_math(self, x, y, center_x, center_y, screen_width=640, screen_height=480):
        # Normalized direction vector (x, y, z)
        dir_x = (x - center_x) / 100.0 
        dir_y = (y - center_y) / 100.0
        return np.array([dir_x, dir_y, 1.0])

    def update_and_average_point(self, point_list, new_point, N):
        point_list.append(new_point)
        if len(point_list) > N: point_list.pop(0)
        if not point_list: return None
        avg_x = int(np.mean([p[0] for p in point_list]))
        avg_y = int(np.mean([p[1] for p in point_list]))
        return (avg_x, avg_y)

    def find_line_intersection(self, ellipse1, ellipse2):
        (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
        (cx2, cy2), (_, minor_axis2), angle2 = ellipse2
        angle1_rad, angle2_rad = np.deg2rad(angle1), np.deg2rad(angle2)
        dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
        dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)
        A = np.array([[dx1, -dx2], [dy1, -dy2]])
        B = np.array([cx2 - cx1, cy2 - cy1])
        if np.linalg.det(A) == 0: return None
        t1, t2 = np.linalg.solve(A, B)
        return (int(cx1 + t1 * dx1), int(cy1 + t1 * dy1))

    def compute_average_intersection(self, ray_lines, N, M):
        if len(ray_lines) < 2 or N < 2: return (0, 0)
        selected_lines = random.sample(ray_lines, min(N, len(ray_lines)))
        intersections = []
        for i in range(len(selected_lines) - 1):
            line1, line2 = selected_lines[i], selected_lines[i+1]
            if abs(line1[2] - line2[2]) >= 2:
                intersection = self.find_line_intersection(line1, line2)
                if intersection:
                    self.stored_intersections.append(intersection)
        if len(self.stored_intersections) > M:
            self.stored_intersections = self.stored_intersections[-M:]
        if not self.stored_intersections: return None
        avg_x = np.mean([pt[0] for pt in self.stored_intersections])
        avg_y = np.mean([pt[1] for pt in self.stored_intersections])
        return (int(avg_x), int(avg_y))

    def process_frame(self, frame):
        frame = self.crop_to_aspect_ratio(frame)
        if frame is None: return None, None
        
        darkest_point = self.get_darkest_area(frame)
        if darkest_point is None: return frame, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        val = gray[darkest_point[1], darkest_point[0]]
        
        bin_img = self.apply_binary_threshold(gray, val, 15)
        bin_img = self.mask_outside_square(bin_img, darkest_point, 250)
        
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(bin_img, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reduced = self.filter_contours(contours, 1000, 3)

        center_x, center_y = None, None

        if reduced and len(reduced[0]) > 5:
            final_ellipse = cv2.fitEllipse(reduced[0])
            center_x, center_y = map(int, final_ellipse[0])
            self.ray_lines.append(final_ellipse)
            if len(self.ray_lines) > self.max_rays:
                self.ray_lines.pop(0)
            
            # Draw on Eye Frame
            cv2.ellipse(frame, final_ellipse, (0, 255, 0), 2)

        model_center = self.compute_average_intersection(self.ray_lines, 5, 1500)
        model_center_avg = (320, 240)
        if model_center:
            model_center_avg = self.update_and_average_point(self.model_centers, model_center, 200) or (320, 240)
        
        if model_center_avg[0] == 320: model_center_avg = self.prev_model_center_avg
        else: self.prev_model_center_avg = model_center_avg

        # Draw Center
        cv2.circle(frame, model_center_avg, 5, (255, 0, 0), -1)

        gaze_vector = None
        if center_x is not None:
            cv2.line(frame, model_center_avg, (center_x, center_y), (0, 255, 255), 2)
            gaze_vector = self.compute_gaze_vector_math(center_x, center_y, model_center_avg[0], model_center_avg[1])

        return frame, gaze_vector

# ==========================================
#           PART 2: THE AI WORKER
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
#           PART 3: TRACKING HELPERS
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

def map_gaze_to_screen(gaze_vector, world_shape):
    h, w, _ = world_shape
    
    # FORCE the gaze to be in the center of the screen
    screen_x = int(w / 2)
    screen_y = int(h / 2)
    
    return (screen_x, screen_y)
# ==========================================
#           PART 4: MAIN APP
# ==========================================
def run_app(world_source, eye_source):
    # 1. Start AI
    t = threading.Thread(target=ai_worker_thread, daemon=True)
    t.start()

    # 2. Start Eye Processor
    eye_processor = EyeGazeProcessor()

    # 3. Load YOLO & Embedder
    print(f"[INFO] Loading YOLO: {YOLO_MODEL}...")
    yolo = YOLO(YOLO_MODEL)
    embedder, transform = make_embedder()

    # 4. Tracking State
    recent_ids = deque(maxlen=LAST_N_IDS) 
    id_embeddings = {}                    
    next_id = 0

    # 5. Open Cameras (Supports video looping)
    
    # Helper to open source (Video file or Int ID)
    def open_source(src):
        if str(src).isdigit():
            return cv2.VideoCapture(int(src))
        return cv2.VideoCapture(src)

    cap_world = open_source(world_source)
    cap_eye = open_source(eye_source)

    if not cap_world.isOpened() or not cap_eye.isOpened():
        print(f"[ERROR] Could not open sources. World: {world_source}, Eye: {eye_source}")
        return
    
    print("\n[INFO] READY! Using Video Loop for Eye.\n")

    while True:
        ret_w, frame_world = cap_world.read()
        ret_e, frame_eye = cap_eye.read()

        # LOOPING LOGIC: If eye video ends, restart it
        if not ret_e and str(eye_source).isdigit() == False:
            cap_eye.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_e, frame_eye = cap_eye.read()

        # If World video ends (if it's a file), restart it
        if not ret_w and str(world_source).isdigit() == False:
            cap_world.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_w, frame_world = cap_world.read()

        if not ret_w or not ret_e: 
            break

        # --- PROCESS EYE GAZE ---
        frame_eye_processed, gaze_vector = eye_processor.process_frame(frame_eye)
        
        has_gaze = False
        gx, gy = (0, 0)
        
        if gaze_vector is not None and frame_world is not None:
            gx, gy = map_gaze_to_screen(gaze_vector, frame_world.shape)
            has_gaze = True

        # --- PROCESS WORLD (YOLO) ---
        if frame_world is not None:
            results = yolo(frame_world, imgsz=IMG_SZ, conf=CONF_THRESH, verbose=False)
            
            detections = [] 
            crops = []
            
            for r in results:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    if cls_id in IGNORED_CLASSES: continue

                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    if (x2-x1) < 20 or (y2-y1) < 20: continue
                    
                    detections.append((x1, y1, x2, y2))
                    crops.append(frame_world[y1:y2, x1:x2])

            # --- PROCESS RE-ID / TRACKING ---
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

            # --- VISUALIZATION & GAZE INTERACTION ---
            
            # Draw Gaze Point
            if has_gaze:
                cv2.circle(frame_world, (gx, gy), 12, (0, 0, 255), 2) # Red Circle
                cv2.line(frame_world, (gx-10, gy), (gx+10, gy), (0,0,255), 1)
                cv2.line(frame_world, (gx, gy-10), (gx, gy+10), (0,0,255), 1)

            for (x1, y1, x2, y2), tid in zip(detections, current_frame_assignments):
                text = ai_results.get(tid, "")
                
                color = (200, 200, 200) # Default Gray
                
                # CHECK INTERSECTION WITH GAZE
                if has_gaze:
                    if x1 < gx < x2 and y1 < gy < y2:
                        color = (0, 255, 0) # Green (Looking at it)
                        
                        # Increment Dwell Counter
                        gaze_dwell_counter[tid] = gaze_dwell_counter.get(tid, 0) + 1
                        
                        # Draw Loading Bar
                        dwell_ratio = min(1.0, gaze_dwell_counter[tid] / DWELL_THRESHOLD)
                        bar_w = int((x2-x1) * dwell_ratio)
                        cv2.rectangle(frame_world, (x1, y2+5), (x1+bar_w, y2+10), (0, 255, 0), -1)

                        # TRIGGER AI
                        if gaze_dwell_counter[tid] == DWELL_THRESHOLD:
                            if tid not in ai_results: # Don't re-trigger if already done
                                ai_results[tid] = "Thinking..."
                                
                                # Add padding for context
                                h, w, _ = frame_world.shape
                                p = CONTEXT_PADDING
                                cx1, cy1 = max(0, x1-p), max(0, y1-p)
                                cx2, cy2 = min(w, x2+p), min(h, y2+p)
                                crop = frame_world[cy1:cy2, cx1:cx2].copy()
                                
                                ollama_queue.put((tid, crop))
                    else:
                        gaze_dwell_counter[tid] = 0 # Reset if look away

                # Draw Box
                cv2.rectangle(frame_world, (x1,y1), (x2,y2), color, 2)
                
                # Show ID
                label = f"ID {tid}"
                cv2.putText(frame_world, label, (x1, max(15, y1-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show AI Text
                if text and text != "Thinking...":
                    cv2.putText(frame_world, text, (x1, y2 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                elif text == "Thinking...":
                    cv2.putText(frame_world, "Analyzing...", (x1, y2 + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("World Gaze AI", frame_world)
        
        if frame_eye_processed is not None:
            cv2.imshow("Eye Tracker", frame_eye_processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_world.release()
    cap_eye.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="0", help="ID or path for world camera")
    parser.add_argument("--eye", default="1", help="ID or path for eye camera")
    args = parser.parse_args()
    
    run_app(args.world, args.eye)