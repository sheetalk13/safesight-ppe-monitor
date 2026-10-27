"""
vision_module.py
----------------
Handles the YOLOv8 PPE detection model.
Separated from the main app for easier retraining/swapping of weights.
"""

from ultralytics import YOLO
from pathlib import Path
from huggingface_hub import hf_hub_download
import torch
from PIL import Image, ImageDraw
import cv2
from config import BOX_THRESHOLD

# Set device once for the module
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VisionModule:
    def __init__(self, repo_id="killuminati1/construction-ppe-yolov8", filename="best.pt"):
        print(f"[Vision] Loading YOLOv8 from {repo_id}...")

        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = YOLO(weights_path).to(DEVICE)
        
        # 1. Map 'helmet' to 'hard_hat' directly in the names dictionary.
        # This ensures result.plot() automatically draws the correct label text.
        self.names = self.model.names
        for k, v in list(self.names.items()):
            if v.lower() == 'helmet':
                self.names[k] = 'hard_hat'
            elif v.lower() == 'no-helmet':
                self.names[k] = 'no-hard_hat'
                
        print(f"[Vision] Model loaded on {DEVICE}")

    def get_detections(self, pil_img):
        results_list = self.model(pil_img, conf=BOX_THRESHOLD)  # returns list of Results
        result = results_list[0]            # single image inference

        # If there are detections
        if result.boxes is not None and len(result.boxes) > 0:
            
            # 2. Filter out conflicting negative classes (e.g., 'no-boots' vs 'boots')
            # ONLY when they spatially overlap on the exact same location (IoU > 0.3 or IoA > 0.4).
            boxes_coords = result.boxes.xyxy.tolist()
            keep_indices = []
            for i, cls in enumerate(result.boxes.cls):
                label = self.names[int(cls)].lower()
                
                if label.startswith('no-'):
                    positive_version = label[3:]  # e.g., 'no-boots' -> 'boots', 'no-hard_hat' -> 'hard_hat'
                    neg_box = boxes_coords[i]
                    conflicting = False
                    
                    for j, other_cls in enumerate(result.boxes.cls):
                        if i == j:
                            continue
                        other_label = self.names[int(other_cls)].lower()
                        if other_label == positive_version:
                            pos_box = boxes_coords[j]
                            ix1 = max(neg_box[0], pos_box[0])
                            iy1 = max(neg_box[1], pos_box[1])
                            ix2 = min(neg_box[2], pos_box[2])
                            iy2 = min(neg_box[3], pos_box[3])
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                a_neg = (neg_box[2] - neg_box[0]) * (neg_box[3] - neg_box[1])
                                a_pos = (pos_box[2] - pos_box[0]) * (pos_box[3] - pos_box[1])
                                union = a_neg + a_pos - inter
                                iou = inter / union if union > 0 else 0
                                ioa = inter / min(a_neg, a_pos) if min(a_neg, a_pos) > 0 else 0
                                if iou > 0.3 or ioa > 0.4:
                                    conflicting = True
                                    break
                    if conflicting:
                        continue  # Positive confirmed on this exact object; drop conflicting negative
                        
                keep_indices.append(i)

            # Apply the filter to the result boxes so overlapping bad boxes aren't drawn
            result.boxes = result.boxes[keep_indices]

            # Annotate the image using the filtered boxes
            annotated_img = result.plot()  # returns NumPy array (BGR)

            # Convert boxes to list of dicts for MLP
            detections = []
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x0, y0, x1, y1 = box.tolist()
                detections.append({
                    "label": self.names[int(cls)].lower(),
                    "confidence": float(conf),
                    "box": [x0, y0, x1, y1]
                })

            # Convert annotated image to PIL (for Gradio)
            from PIL import Image
            annotated_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))

            return annotated_img, detections
        else:
            # No detections
            return pil_img, []