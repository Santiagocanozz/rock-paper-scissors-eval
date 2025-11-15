def normalize_label(label: str) -> str:
    """Convierte labels YOLO → Español."""
    mapping = {
        "rock": "Piedra",
        "paper": "Papel",
        "scissors": "Tijera",
        "undecided": "Indeciso"
    }
    return mapping.get(label.lower(), label)


def decide_winner(a: str, b: str) -> str:
    """Determina el ganador según las reglas del juego."""
    if a == "undecided" or b == "undecided":
        return "undecided"

    if a == b:
        return "tie"

    rules = {
        ("rock", "scissors"): "player_a",
        ("scissors", "paper"): "player_a",
        ("paper", "rock"): "player_a",
        
        ("scissors", "rock"): "player_b",
        ("paper", "scissors"): "player_b",
        ("rock", "paper"): "player_b",
    }

    return rules.get((a, b), "undecided")
