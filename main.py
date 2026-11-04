from ultralytics import YOLO

model = YOLO("yolo11n.pt")

image_path = "dataset/images/img_07.jpg"

results = model(image_path)

for result in results:

    for box in result.boxes:

        # Identificação da classe
        class_id = int(box.cls[0])
        class_name = result.names[class_id]

        # Confiança da detecção
        confidence = float(box.conf[0])

        print("Veículo:", class_name)
        print("Confiança:", confidence)
        print("-------------------")

    result.show()