# Rock-Paper-Scissors Evaluation System (FastAPI + YOLOv8)

Se generó un pipeline para detección de manos (Piedra, Papel, Tijera), donde el sistema determina automáticamente el ganador y retorna una respuesta estructurada vía API.  
Incluye detección con **YOLOv8**, reglas del juego, bounding boxes, umbral de confianza, validación de dataset, pruebas manuales, pruebas masivas y despliegue con **Docker**.

---

# 1. Objetivo del Proyecto

El sistema permite:

- Detectar dos imágenes de manos (jugador A y jugador B)  
- Predecir: Piedra, Papel o Tijera  
- Evaluar la confianza del modelo  
- Retornar bounding boxes  
- Determinar el ganador según reglas:  
  - Piedra vence a Tijera  
  - Tijera vence a Papel  
  - Papel vence a Piedra  
- Retornar un mensaje JSON estructurado  

**Endpoint principal:**  

<pre style="font-family: monospace; font-size: 14px;">
rock-paper-scissors-eval/
│
├── src/
│   ├── app/
│   │   ├── main.py                    # API FastAPI
│   │   ├── model.py                   # YOLOService
│   │   └── utils.py                   # Lógica del juego
│   │
│   ├── inference/
│   │   └── predict.py                 # Script de predicción local
│   │
│   ├── tests/
│   │   ├── test_api.py                # Pruebas manuales
│   │   └── report_eval_directory.json # Stress test masivo
│
├── train/
│   ├── validate_dataset.py            # Validación del dataset
│   └── train.py                       # Entrenamiento YOLOv8
│
├── data/
│   ├── data.yaml
│   └── dataset YOLO               
│
├── weights/
│   └── best.pt                        # Modelo entrenado final
│
├── requirements.txt
└── Dockerfile
</pre>

# 3. Validación Inicial del Dataset

Antes de entrenar se realizó una validación completa:

- Verificación de `data.yaml`  
- Comprobación imágenes ↔ labels  
- Validación del formato YOLO  
- Detección de clases faltantes  
- Conteo de muestras por clase  
- Revisión del balance del dataset  

**Ejemplo de resultados:**

train: 160 imágenes, 160 labels
valid: 46 imágenes, 46 labels

Clases:
Piedra: 46
Papel: 44
Tijera: 43

VALIDACIÓN COMPLETA


---

# 4. Entrenamiento del Modelo

Entrenamiento realizado con **Ultralytics YOLOv8** utilizando GPU **RTX 4050** para mejorar tiempos y desempeño.

### Augmentación avanzada utilizada:

- mosaic  
- mixup  
- copy-paste  
- rotaciones (±25°)  
- shear  
- flips  
- hsv augment  
- perspective transform  

### Modelo base

yolov8s.pt


(precisión superior a yolov8n)

### Resultados finales

- **mAP50:** 0.92 – 0.96  
- **mAP50-95:** 0.60 – 0.72  
- Precisión estable entre clases  

---

# 5. Arquitectura del Sistema

## 5.1 YOLOService (`src/app/model.py`)

Funciones principales:

- cargar el modelo YOLO  
- ejecutar inferencia  
- seleccionar detección más confiable  
- generar bounding boxes en enteros  
- normalizar coordenadas  
- retornar indecisión si `confidence < threshold`  

**Salida típica:**

[x, y, width, height]


---

## 5.2 Utils (`src/app/utils.py`)

Responsable de:

- Normalizar etiquetas a español:  
  - rock → Piedra  
  - paper → Papel  
  - scissors → Tijera  
- Determinar el ganador según reglas del juego  

---

## 5.3 API FastAPI (`src/app/main.py`)

**Endpoint:**


**Request (multipart/form-data):**

- `player_a`: archivo de imagen  
- `player_b`: archivo de imagen  

**Ejemplo de respuesta:**

```json
{
  "player_a": {
    "prediction": "Piedera",
    "confidence": 0.95,
    "bbox": [120, 80, 210, 230]
  },
  "player_b": {
    "prediction": "Tijera",
    "confidence": 0.88,
    "bbox": [140, 70, 200, 210]
  },
  "winner": "player_a",
  "reason": "Piedra vence a Tijera"
}
```
## 6. Pruebas Manuales

Ubicadas en:

src/tests/test_api.py

Incluye validaciones de:

- Piedra vence a Tijera  
- Papel vence a Piedra  
- Tijera vence a Papel  
- Rock vs Rock → Empate  
- Imagen sin mano → Indecisión  
- Latencia menor a 500 ms  

---

## 7. Prueba Masiva (Stress Test)

Se evaluaron todas las combinaciones A vs B del dataset de validación.

Se midió:

- latencia promedio  
- latencia máxima  
- predicciones más frecuentes  
- indecisiones  
- errores del API  

Reporte generado en:

src/tests/report_eval_directory.json


---

## 8. Docker — Build y Run

**Build**

docker build -t rps-api .


**Run**

docker run -p 8000:8000 rps-api


**Acceder a la documentación**

http://127.0.0.1:8000/docs

---

## 9. Mejoras Futuras

- Exportar modelo a ONNX / TensorRT para acelerar inferencia  
- Mejorar robustez ante manos parcialmente visibles  
- Agregar logging estructurado para auditoría  
- Crear Dockerfile con soporte CUDA  


