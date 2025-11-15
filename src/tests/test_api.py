import requests
import time
import json
import os
from glob import glob

API_URL = "http://127.0.0.1:8000/play"

TEST_IMAGES = {
    "rock": "data/valid/images/egohands-public-1622127386865_png_jpg.rf.2aa690a3f06c3a28d05b5b74bb1511ba.jpg",
    "paper": "data/valid/images/egohands-public-1624298528652_png_jpg.rf.9587f1aebc15d6bf9af1cb74277cf552.jpg",
    "scissors": "data/valid/images/egohands-public-1625070646143_png_jpg.rf.0e61e0712aa02d1560234dfcc1bde5a7.jpg"
}

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró la imagen de prueba: {path}")


def send_request(img_a, img_b):
    files = {
        "player_a": open(img_a, "rb"),
        "player_b": open(img_b, "rb")
    }
    start = time.time()
    response = requests.post(API_URL, files=files)
    latency = time.time() - start

    return response, latency


def test_case(title, img_a, img_b):
    print("=" * 60)
    print(f"[TEST] {title}")
    print("-" * 60)

    check_file(img_a)
    check_file(img_b)

    response, latency = send_request(img_a, img_b)

    try:
        data = response.json()
    except Exception:
        print("[ERROR] La API devolvió un formato no JSON")
        print(response.text)
        return

    print(f"→ Status Code: {response.status_code}")
    print(f"→ Latencia: {latency:.4f} s")
    print(f"→ Respuesta:")
    print(json.dumps(data, indent=4))

    if latency > 0.5:
        print("La latencia supera los 500 ms (requisito de la prueba).")
    print("\n")


def run_all_manual_test():

    test_case(
        "Rock vence a Scissors",
        TEST_IMAGES["rock"],
        TEST_IMAGES["scissors"]
    )

    test_case(
        "Paper vence a Rock",
        TEST_IMAGES["paper"],
        TEST_IMAGES["rock"]
    )

    test_case(
        "Scissors vence a Paper",
        TEST_IMAGES["scissors"],
        TEST_IMAGES["paper"]
    )

    test_case(
        "Empate — Rock vs Rock",
        TEST_IMAGES["rock"],
        TEST_IMAGES["rock"]
    )

    test_case(
        "Indecisión — Imagen sin mano",
        "data/valid\images/Screen-Shot-2022-02-14-at-1-09-07-PM_png.rf.36ab76b4e51ee42a9181d2f5e3e8d75b.jpg",
        TEST_IMAGES["rock"]
    )


'''
En este escenario se busca evaluar el rendimiento del API ante la carga de multiples 
imagenes
'''
def eval_directory(dir_path):
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"No existe el directorio: {dir_path}")

    images = glob(os.path.join(dir_path, "*.*"))
    if len(images) < 2:
        raise ValueError("Se necesitan al menos 2 imágenes en el directorio.")

    total_pairs = 0
    latencies = []
    errors = 0
    undecided = 0
    predictions = {
        "rock": 0,
        "paper": 0,
        "scissors": 0,
        "undecided": 0
    }

    print(f"/nIniciando evaluación masiva en {dir_path}...")
    print(f"Total de imágenes encontradas: {len(images)}")

    for img_a in images:
        for img_b in images:
            if img_a == img_b:
                continue

            total_pairs += 1
            response, latency = send_request(img_a, img_b)
            latencies.append(latency)

            try:
                data = response.json()
                winner = data.get("winner", "undecided")
            except:
                errors += 1
                continue

            # contar estadísticas básicas
            if winner == "undecided":
                undecided += 1
            else:
                predictions[data["player_a"]["prediction"]] = (
                    predictions.get(data["player_a"]["prediction"], 0) + 1
                )

            print(f"{os.path.basename(img_a)} vs {os.path.basename(img_b)} "
                  f"→ {winner} | {latency:.4f}s")


    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    undecided_rate = undecided / total_pairs

    summary = {
        "total_pairs": total_pairs,
        "avg_latency_sec": round(avg_latency, 4),
        "max_latency_sec": round(max_latency, 4),
        "errors": errors,
        "undecided": undecided,
        "undecided_rate": round(undecided_rate, 4),
        "predictions_summary": predictions
    }

    print("\n===== RESUMEN GENERAL (DIRECTORIO) =====")
    print(json.dumps(summary, indent=4))

    with open("src/tests/report_eval_directory.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nReporte guardado en src/tests/report_eval_directory.json\n")

    return summary

if __name__ == "__main__":
    run_all_manual_test()
    eval_directory("data/valid/images")
