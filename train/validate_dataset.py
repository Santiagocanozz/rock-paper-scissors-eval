"""
En este caso por temas de no conocimiento del dataset, decido agregar 
una breve validación antes de entrenar.
"""

import os
import yaml
from glob import glob
from collections import Counter
from pathlib import Path


def loda_dataset_yml(path_yml_dataset):
    try:
        with open(path_yml_dataset, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo YAML: {path_yml_dataset}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error al procesar el YAML {path_yml_dataset}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error inesperado leyendo el YAML {path_yml_dataset}: {str(e)}")


def validate_yaml(data_yaml):
    print(f"Validando dataset.yaml en: {data_yaml}")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"dataset.yaml no existe en la ruta: {data_yaml}")

    data = loda_dataset_yml(data_yaml)

    required_fields = ["train", "val", "names"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"dataset.yaml está incompleto. Falta el campo '{field}'")

    names = data.get("names")
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    elif not isinstance(names, dict):
        raise ValueError("El campo 'names' debe ser una lista o un diccionario.")
    data["names"] = names

    print("dataset.yaml válido")
    return data


def validate_list_images_and_labels(base_path, split_for_type_file):
    image_folder = os.path.join(base_path, split_for_type_file, "images")
    label_folder = os.path.join(base_path, split_for_type_file, "labels")

    try:
        list_images = sorted(glob(os.path.join(image_folder, "*.*g")))
        labels = sorted(glob(os.path.join(label_folder, "*.txt")))
    except Exception as e:
        raise RuntimeError(f"Error leyendo archivos en {image_folder} o {label_folder}: {str(e)}")

    if not list_images:
        raise ValueError(f"No se encontraron imágenes en: {image_folder}")

    if not labels:
        raise ValueError(f"No se encontraron labels en: {label_folder}")

    print(f"{split_for_type_file}: {len(list_images)} imágenes, {len(labels)} labels")

    try:
        missing = []
        for image in list_images:
            filename = Path(image).stem
            label = os.path.join(label_folder, f"{filename}.txt")
            if not os.path.exists(label):
                missing.append(image)

        if missing:
            count_images = len(missing)
            raise ValueError(f"Faltan {count_images} labels correspondientes a imágenes en {split_for_type_file}")

    except Exception as e:
        raise RuntimeError(f"Error validando correspondencia imagen-label en {split_for_type_file}: {str(e)}")

    print(f"{split_for_type_file}: correspondencia imagen-label correcta")
    return labels


def validate_yolo_format(labels, class_names):
    class_counts = Counter()

    try:
        for label_path in labels:
            with open(label_path, "r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split()

                    if len(parts) != 5:
                        raise ValueError(f"Formato inválido en {label_path}. Cada línea debe tener 5 valores.")

                    cls = int(parts[0])

                    if cls not in class_names:
                        valid = list(class_names.keys())
                        raise ValueError(f"Clase inválida '{cls}' en {label_path}. Clases válidas: {valid}")

                    class_counts[cls] += 1

    except FileNotFoundError:
        raise FileNotFoundError(f"No se pudo leer el archivo de label: {label_path}")
    except ValueError as e:
        raise ValueError(str(e))
    except Exception as e:
        raise RuntimeError(f"Error inesperado procesando el label {label_path}: {str(e)}")

    print("Formato YOLO válido")

    print("Distribución de clases:")
    for class_id, count in class_counts.items():
        name = class_names[class_id]
        print(f"{name}: {count} muestras")

    if min(class_counts.values()) == 0:
        raise ValueError("El dataset NO está balanceado. Hay clases sin muestras.")

    return class_counts


def main():
    base_path = "data"
    data_yaml = os.path.join(base_path, "data.yaml")

    try:
        data = validate_yaml(data_yaml)
        train_labels = validate_list_images_and_labels(base_path, "train")
        val_labels = validate_list_images_and_labels(base_path, "valid")
        validate_yolo_format(train_labels, data["names"])


    except Exception as e:
        print("ERROR DURANTE LA VALIDACIÓN DEL DATASET:")
        print(str(e))
        print("Corrija el problema antes de entrenar.")
        return


if __name__ == "__main__":
    main()
