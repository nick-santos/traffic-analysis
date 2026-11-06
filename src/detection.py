from pathlib import Path


def detect_vehicles(model, image_folder):
    # Definir as categorias de veículos
    vehicle_classes = ["car", "motorcycle", "bus", "truck"]

    images = sorted(Path(image_folder).glob("*"))

    detections = []

    for image_path in images:

        print(f"\nAnalisando: {image_path.name}")

        results = model(str(image_path))

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])

                if class_name in vehicle_classes:

                    detection = {
                        "image": image_path.name,
                        "category": class_name,
                        "confidence": confidence
                    }

                    detections.append(detection)

    return detections


def save_detection_images(model, results_images_folder, image_folder):
    images = sorted(Path(image_folder).glob("*"))

    for image_path in images:
        results = model(str(image_path))

        for result in results:
            output_path = results_images_folder / image_path.name
            result.save(filename=str(output_path))
