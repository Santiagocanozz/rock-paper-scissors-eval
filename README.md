# Rock-Paper-Scissors Evaluation System (FastAPI + YOLOv8)

Se genero un pipeline para detección de manos (Piedra, Papel, Tijera), donde se genera
una decisión automática del ganador y respuesta estructurada vía API.  
En este caso se implementa detección con **YOLOv8**, reglas de juego, bounding boxes, umbral de confianza, validación de dataset, pruebas manuales, pruebas masivas y despliegue con **Docker**.

---

# 1. Objetivo del Proyecto

Construir un sistema que:
- Detecta dos imágenes de manos (jugador A y B)
- Predice: Piedra, Papel o Tijera
- Evalúa la confianza del modelo
- Retorna bounding boxes
- Determina el ganador según reglas:
  - Piedra vence a Tijera
  - Tijera vence a Papel
  - Papel vence a Piedra
- Retorna mensaje estructurado.

El API expone un único endpoint:
Metodo de tipo POST
POST /play

Estructura de implementación: 

rock-paper-scissors-eval/
│
├── src/
│ ├── app/
│ │ ├── main.py # API FastAPI
│ │ ├── model.py # YOLOService
│ │ └── utils.py # Lógica del juego
│ │
│ ├── inference/
│ │ └── predict.py # Script de predicción local
│ │
│ ├── tests/
│ ├── test_api.py # Pruebas manuales
│ └── report_eval_directory.json # Stress test masivo
│
├── train/
│ ├── validate_dataset.py # Validación del dataset
│ └── train.py # Entrenamiento YOLOv8
│
├── data/
│ ├── data.yaml
│ └── (dataset YOLO)
│
├── weights/
│ └── best.pt
│
├── requirements.txt
└── Dockerfile

¿Como se ejecuto desde el principio la prueba?
Antes de entrenar, se realizó una validación completa:

- Verificación de `data.yaml`
- Comprobación imágenes ↔ labels uno a uno
- Validación del formato YOLO
- Detección de clases faltantes
- Conteo de muestras por clase
- Revisión de balance del dataset

Ejemplo de salida: 

train: 160 imágenes, 160 labels
valid: 46 imágenes, 46 labels
Clases:
Piedra: 46
Papel: 44
Tijera: 43
VALIDACIÓN COMPLETA

Luego de esto se genero:

Entrenamiento de modelo con Ultralytics y se genero entrenamiento con GPU RTX4050 debido a limites de tiempo y mejora es respuesta.

Finalmente se implementaron tecnicas en el proceso de entrenamiento que fuesen de mayor apoyo en la busqueda
de un mejor rendimiento

Augmentación avanzada usada:
mosaic
mixup
copy-paste
rotaciones (±25°)
shear
flips
hsv augment
perspective transform

Modelo implementado: 
yolov8s.pt (mejor precisión que yolov8n)

Resultados finales:
mAP50: 0.92 – 0.96
mAP50-95: 0.60 – 0.72
Precisión muy estable entre clases

En si la arquitectura del sistema:
YOLOService (src/app/model.py)
En esta se ejecuta:

cargar el modelo YOLO
hacer inferencia
seleccionar la detección más confiable
generar bounding boxes con valores enteros
normalizar y limpiar bbox dentro del tamaño de la imagen
retornar indecisión si la confianza < threshold

Se obtiene como salida: 
[x, y, width, height]

Utils (src/app/utils.py)

Encargado de:
normalizar etiquetas a español:
rock → Piedra
paper → Papel
scissors → Tijera
decidir el ganador según reglas del juego

API (src/app/main.py)
Endpoint principal:
POST /play

Request (multipart/form-data):
player_a: archivo imagen
player_b: archivo imagen

Ejemplo de salida: 

{
  "player_a": {
    "prediction": "Piedra",
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

Pruebas manuales: 

src/tests/test_api.py

Incluye:

✔ Rock vence a Scissors
✔ Paper vence a Rock
✔ Scissors vence a Paper
✔ Rock vs Rock → Empate
✔ Imagen sin mano → Indecisión
✔ Medición de latencia (< 500 ms)


Prueba Masiva (Stress Test)
Se evalúan todas las combinaciones A vs B del conjunto data/valid/images/.
mide latencia promedio
latencia máxima
predicciones más frecuentes
indecisiones
errores de API
genera reporte JSON
src/tests/report_eval_directory.json


Docker — Build y Run
Build:
docker build -t rps-api .

Run:
docker run -p 8000:8000 rps-api

Acceder:
http://127.0.0.1:8000/docs

Mejoras Futuras

Exportar modulo a ONNX/TensorRT para acelerar inferencia
Mejorar robustez ante manos parcialmente visibles
Agregar logging estructurado para auditoría
Dockerfile con soporte CUDA


Conclusión

El sistema cumple completamente con los requerimientos:
modelo entrenado con excelente precisión
validación robusta del dataset
API estable, rápida y documentada
bounding boxes limpios y seguros
lógica del juego implementada perfectamente
entregable completo y dockerizado
pruebas exhaustivas manuales y masivas