# # yolo_appearance_last10_resnet50.py
# """
# Appearance-only, no-motion tracker using ResNet50 embeddings (2048-d).
# - Reuses IDs from last-N history by cosine similarity of ResNet50 embeddings.
# - Optionally ignores classes (e.g., person, dining table).

# Usage:
#   python yolo_appearance_last10_resnet50.py --video sample.mp4
#   python yolo_appearance_last10_resnet50.py --video 0

# Requires:
#   pip install ultralytics opencv-python torch torchvision numpy
# """
# import argparse
# import time
# import cv2
# import numpy as np
# from collections import deque
# from ultralytics import YOLO
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# import torchvision.models as models
# from torchvision.models import ResNet50_Weights
# import io
# import warnings

# # -------- CONFIG ----------
# YOLO_PREFERRED = "yolo11n.pt"
# YOLO_FALLBACK = "yolov8n.pt"
# CONF_THRESH = 0.35
# IMG_SZ = 320

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("[INFO] device:", DEVICE)

# # note: stronger embeddings -> use a slightly higher threshold
# SIMILARITY_THRESHOLD = 0.6   # cosine similarity threshold (set higher for ResNet50)
# LAST_N_IDS = 10               # keep the last 10 IDs in history
# MIN_BOX_SIDE = 12

# # Classes to ignore (case-sensitive names coming from the model's names dict)
# IGNORED_CLASSES = {"person", "dining table", "table"}

# API_URL = ""     # set your API endpoint or leave empty (stub)
# API_KEY = ""     # optional
# JPEG_QUALITY = 80
# # --------------------------

# def load_yolo():
#     try:
#         m = YOLO(YOLO_PREFERRED)
#         print(f"[INFO] Loaded {YOLO_PREFERRED}")
#         return m
#     except Exception as e:
#         print(f"[WARN] Couldn't load {YOLO_PREFERRED}: {e}. Falling back to {YOLO_FALLBACK}")
#         m = YOLO(YOLO_FALLBACK)
#         print(f"[INFO] Loaded {YOLO_FALLBACK}")
#         return m

# def make_resnet50_embedder(device):
#     """
#     Returns:
#       feature_extractor: model that maps (1,3,224,224) -> (1,2048,1,1)
#       transform: preprocessing
#       feat_dim: 2048
#     """
#     # load pretrained ResNet50 (modern weights enum)
#     resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
#     resnet.eval()
#     # remove final fc layer & keep avgpool output via taking children except last fc
#     feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
#     feature_extractor.eval()

#     transform = T.Compose([
#         T.ToPILImage(),
#         T.Resize((224,224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#     ])
#     feat_dim = 2048
#     return feature_extractor, transform, feat_dim

# def get_embedding_resnet(feature_extractor, transform, crop_bgr, device):
#     """
#     crop_bgr: HxWx3 BGR uint8 -> returns 1D numpy array of length feat_dim (L2-normalized)
#     """
#     if crop_bgr is None or crop_bgr.size == 0:
#         raise RuntimeError("empty crop for embedding")
#     crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#     x = transform(crop_rgb).unsqueeze(0).to(device)  # 1,C,H,W
#     with torch.no_grad():
#         feats = feature_extractor(x)  # [1, 2048, 1, 1]
#         pooled = feats.squeeze()      # [2048] or shape (2048,)
#         if pooled.ndim == 0:
#             pooled = pooled.view(-1)
#         vec = pooled.cpu().float().numpy()
#         vec = vec / (np.linalg.norm(vec) + 1e-8)
#         return vec

# def cosine_sim(a, b):
#     a = np.asarray(a, dtype=np.float32)
#     b = np.asarray(b, dtype=np.float32)
#     if b.ndim == 1:
#         denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
#         return float(np.dot(a, b) / denom)
#     a_norm = a / (np.linalg.norm(a) + 1e-8)
#     b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
#     return (b_norm @ a_norm)

# def encode_jpeg(img_bgr, q=80):
#     ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
#     if not ok:
#         raise RuntimeError("jpeg encode failed")
#     return buf.tobytes()

# def send_to_api_stub(jpeg_bytes, track_id):
#     if not API_URL:
#         print(f"[API-STUB] would send track {track_id} ({len(jpeg_bytes)} bytes)")
#         return {"status":"skipped"}
#     files = {"image": ("obj.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}
#     headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
#     try:
#         import requests
#         r = requests.post(API_URL, files=files, headers=headers, timeout=8)
#         return r.json()
#     except Exception as e:
#         print("[API ERROR]", e)
#         return {"error": str(e)}

# def run(video_source):
#     yolo = load_yolo()
#     feat_extractor, transform, feat_dim = make_resnet50_embedder(DEVICE)
#     print(f"[INFO] embed feature dim = {feat_dim} (ResNet50)")

#     def get_class_name(model_obj, cls_id):
#         try:
#             names = getattr(model_obj, "names", None) or (getattr(model_obj, "model", None) and getattr(model_obj.model, "names", None))
#         except Exception:
#             names = None
#         if not names:
#             return str(int(cls_id))
#         return names.get(int(cls_id), str(int(cls_id)))

#     recent_ids = deque(maxlen=LAST_N_IDS)
#     id_to_embedding = {}
#     id_to_bbox = {}
#     next_id = 0

#     cap = cv2.VideoCapture(video_source if str(video_source) != "0" else 0)
#     if not cap.isOpened():
#         print("[ERROR] cannot open video source:", video_source)
#         return

#     frame_idx = 0
#     prev_time = time.time()
#     print("[INFO] Starting. q to quit, r to reset history")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[INFO] End of video or cannot read frame.")
#             break
#         frame_idx += 1

#         # YOLO detection
#         results = yolo(frame, imgsz=IMG_SZ, conf=CONF_THRESH, verbose=False)
#         detections = []  # (x1,y1,x2,y2,cls_name)
#         crops = []

#         for r in results:
#             for b in r.boxes:
#                 conf = float(b.conf[0]) if hasattr(b.conf, "__getitem__") else float(b.conf)
#                 if conf < CONF_THRESH:
#                     continue
#                 try:
#                     cls_id = int(b.cls[0]) if hasattr(b.cls, "__getitem__") else int(b.cls)
#                 except Exception:
#                     cls_id = -1
#                 cls_name = get_class_name(yolo, cls_id)

#                 if cls_name in IGNORED_CLASSES:
#                     continue

#                 x1,y1,x2,y2 = map(int, b.xyxy[0])
#                 x1, y1 = max(0,x1), max(0,y1)
#                 x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
#                 if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
#                     continue
#                 detections.append((x1, y1, x2, y2, cls_name))
#                 crops.append(frame[y1:y2, x1:x2].copy())

#         # compute embeddings for all crops using ResNet50 extractor
#         embeddings = []
#         for crop in crops:
#             try:
#                 emb = get_embedding_resnet(feat_extractor, transform, crop, DEVICE)
#                 embeddings.append(emb)
#             except Exception as e:
#                 warnings.warn(f"embedding failed for a crop: {e}")
#                 embeddings.append(None)

#         assigned_ids = []

#         for i, det in enumerate(detections):
#             x1, y1, x2, y2, cls_name = det
#             emb = embeddings[i] if i < len(embeddings) else None
#             assigned = None
#             if emb is not None and len(recent_ids) > 0:
#                 ref_ids = list(recent_ids)
#                 valid_ref_ids = [rid for rid in ref_ids if rid in id_to_embedding]
#                 if len(valid_ref_ids) > 0:
#                     ref_embs = np.vstack([id_to_embedding[rid] for rid in valid_ref_ids])
#                     sims = cosine_sim(emb, ref_embs)
#                     best_idx = int(np.argmax(sims))
#                     best_sim = float(sims[best_idx])
#                     if best_sim >= SIMILARITY_THRESHOLD:
#                         assigned = valid_ref_ids[best_idx]
#                         print(f"[REUSE] frame {frame_idx}: id {assigned} | det {i} | class={cls_name} (sim={best_sim:.3f})")
#             if assigned is None:
#                 assigned = next_id
#                 next_id += 1
#                 if emb is not None:
#                     id_to_embedding[assigned] = emb.copy()
#                 else:
#                     id_to_embedding[assigned] = np.zeros((feat_dim,), dtype=np.float32)
#                 recent_ids.append(assigned)
#                 print(f"[NEW] frame {frame_idx}: id {assigned} | det {i} | class={cls_name}")
#             id_to_bbox[assigned] = (x1, y1, x2, y2)
#             assigned_ids.append(assigned)

#         # Draw annotations
#         vis = frame.copy()
#         for (det, tid) in zip(detections, assigned_ids):
#             x1, y1, x2, y2, cls_name = det
#             color = (0,200,0) if tid in recent_ids else (0,120,255)
#             cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
#             cv2.putText(vis,
#                         f"id:{tid} {cls_name}",
#                         (x1, max(15,y1-6)),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         color,
#                         2)

#         # recent ids text + FPS
#         txt = "recent_ids: " + ",".join(map(str, list(recent_ids)))
#         cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)
#         now_time = time.time()
#         fps = 1.0 / (now_time - prev_time) if (now_time - prev_time) > 0 else 0.0
#         prev_time = now_time
#         cv2.putText(vis, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

#         cv2.imshow("Appearance last-N (ResNet50) (q quit, r reset)", vis)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         if key == ord('r'):
#             recent_ids.clear()
#             id_to_embedding.clear()
#             id_to_bbox.clear()
#             print("[ACTION] history reset")

#     cap.release()
#     cv2.destroyAllWindows()
#     print("[INFO] Done.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--video", type=str, default="0",
#                         help="Path to video file or '0' for webcam")
#     args = parser.parse_args()
#     run(args.video)

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
print(DEVICE)