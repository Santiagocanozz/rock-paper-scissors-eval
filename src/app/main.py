'''
 Piedra vence a Tijera
 • Tijera vence a Papel
 • Papel vence a Piedra

'''

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .model import YOLOService
from .utils import decide_winner, normalize_label

app = FastAPI(title="Rock Paper Scissors API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    yolo_model = YOLOService("weights/best.pt", threshold=0.4)
except Exception as e:
    raise RuntimeError(f"No se pudo inicializar el servicio YOLO: {str(e)}")


@app.post("/play")
async def play(player_a: UploadFile = File(...), player_b: UploadFile = File(...)):
    if not player_a or not player_b:
        raise HTTPException(status_code=400, detail="Se requieren player_a y player_b.")

    try:
        pred_a, conf_a, bbox_a = yolo_model.predict(player_a, return_conf=True, return_bbox=True)
        pred_b, conf_b, bbox_b = yolo_model.predict(player_b, return_conf=True, return_bbox=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        winner = decide_winner(pred_a, pred_b)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    na = normalize_label(pred_a)
    nb = normalize_label(pred_b)

    if winner == "player_a":
        reason = f"{na} vence a {nb}"
    elif winner == "player_b":
        reason = f"{nb} vence a {na}"
    elif winner == "tie":
        reason = "Ambos jugadores hicieron la misma jugada"
    else:
        reason = "No se pudo determinar la jugada con suficiente confianza"

    return {
    "player_a": {
        "prediction": normalize_label(pred_a),
        "confidence": round(float(conf_a), 4),
        "bbox": bbox_a
    },
    "player_b": {
        "prediction": normalize_label(pred_b),
        "confidence": round(float(conf_b), 4),
        "bbox": bbox_b
    },
    "winner": winner,
    "reason": reason
}