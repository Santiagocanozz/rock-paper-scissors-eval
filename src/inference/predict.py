'''
En este script se ejecuta la inferencia basada en el modelo entrenado a través de Ultralytics
Se genera aplicación de NMS aprovechando el recurso de Ultralytics: 
https://docs.ultralytics.com/reference/utils/nms/#ultralytics.utils.nms.TorchNMS

'''


import argparse
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np


def predict(model_path: str, image_path: str, save_path: str = "prediction.jpg", threshold: float = 0.4):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo YOLO: {str(e)}")

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir la imagen: {str(e)}")

    try:
        results = model.predict(source=np.array(img), save=False, verbose=False)
    except Exception as e:
        raise RuntimeError(f"Error durante la inferencia: {str(e)}")

    det = results[0]
    boxes = det.boxes

    if len(boxes) == 0:
        return "undecided", 0.0

    idx = boxes.conf.argmax().item()

    cls_id = int(boxes.cls[idx].item())
    class_name = model.names[cls_id]
    confidence = float(boxes.conf[idx].item())

    if confidence < threshold:
        class_name = "undecided"

    try:
        det.save(save_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo guardar la imagen con la predicción: {str(e)}")

    return class_name, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Predictor")
    parser.add_argument("--image", required=True, help="Ruta de la imagen")
    parser.add_argument("--model", default="weights/best.pt", help="Ruta del modelo YOLO")
    parser.add_argument("--output", default="prediction.jpg", help="Archivo de salida")
    parser.add_argument("--threshold", default=0.4, type=float, help="Umbral de confianza")

    args = parser.parse_args()

    try:
        label, conf = predict(args.model, args.image, args.output, args.threshold)
        print(f"Predicción: {label}")
        print(f"Confianza: {conf:.4f}")
        print(f"Imagen guardada en: {args.output}")
    except Exception as e:
        print(f"Error: {str(e)}")