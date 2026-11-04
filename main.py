from ultralytics import YOLO
from pathlib import Path

# Carregar o modelo
model = YOLO("yolo11n.pt")

# Definir as categorias de veículos
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

vehicle_counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}

# Localizar as imagens
image_folder = Path("dataset/images")
images = list(image_folder.glob("*"))

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

# Resultados

for image_path in images:
    results = model(str(image_path))

    for result in results:
        result.show()

image_name = ""
total_confidence = 0

print("\nDetecções:")

for detection in detections:

    if image_name != detection["image"]:
        image_name = detection["image"]
        print(f"\n{image_name}")

    category = detection["category"]
    vehicle_counts[category] += 1

    total_confidence += detection["confidence"]

    print(detection)

# Totais
print("\nTotal de veículos: ", len(detections))

print("\nTotal de veículos por categoria:")

for category, count in vehicle_counts.items():
    print(f"{category}: {count}")

# Media de confiança
if len(detections) > 0:
    average_confidence = total_confidence / len(detections)
else:
    average_confidence = 0

print(f"\nConfiança média: {average_confidence:.2%}")
