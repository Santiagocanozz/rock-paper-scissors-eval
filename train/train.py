from ultralytics import YOLO
import os


def main():
    dataset_yaml = os.path.join("data", "data.yaml")

    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"No se encontró el archivo: {dataset_yaml}")

    try:
        model = YOLO("yolov8s.pt") 
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo base yolov8s.pt: {str(e)}")

    try:
        model.train(
            data=dataset_yaml,
            epochs=180,
            imgsz=640,
            batch=16,                
            device=0,                
            workers=4,
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=True,
            close_mosaic=15,         
            patience=0,              

            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.1,
            degrees=25,
            shear=12,
            translate=0.12,
            scale=0.7,
            perspective=0.002,
            flipud=0.25,
            fliplr=0.6,
        )
    except Exception as e:
        raise RuntimeError(f"Error durante el entrenamiento: {str(e)}")


if __name__ == "__main__":
    main()