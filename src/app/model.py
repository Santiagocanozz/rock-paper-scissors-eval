import os
from ultralytics import YOLO
from PIL import Image
import numpy as np


class YOLOService:
    def __init__(self, model_path: str, threshold: float = 0.4):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
        
        self.model = YOLO(model_path)
        self.threshold = threshold

    def predict(self, image_file, return_conf=False, return_bbox=False):
        try:
            img = Image.open(image_file.file).convert("RGB")
            w_img, h_img = img.size
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir la imagen: {str(e)}")

        try:
            results = self.model.predict(source=np.array(img), save=False, verbose=False)
        except Exception as e:
            raise RuntimeError(f"Error durante la inferencia: {str(e)}")

        boxes = results[0].boxes
        if len(boxes) == 0:
            if return_conf or return_bbox:
                return ("undecided", 0.0, None)
            return "undecided"

        idx = boxes.conf.argmax().item()

        cls_id = int(boxes.cls[idx].item())
        class_name = self.model.names[cls_id].lower()

        confidence = float(boxes.conf[idx].item())

        x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
        x1 = max(0, min(x1, w_img))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img))
        y2 = max(0, min(y2, h_img))

        if x2 <= x1:
            x2 = x1 + 1
        if y2 <= y1:
            y2 = y1 + 1

        bbox = [
            int(round(x1)),
            int(round(y1)),
            int(round(x2 - x1)),
            int(round(y2 - y1))
        ]

        if confidence < self.threshold:
            return ("undecided", confidence, bbox)

        if return_conf or return_bbox:
            return class_name, confidence, bbox

        return class_name
