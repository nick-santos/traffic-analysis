from ultralytics import YOLO
from pathlib import Path
import pandas as pd

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


# Guadar imagens com Bounding boxes

output_folder = Path("results/detections")
output_folder.mkdir(parents=True, exist_ok=True)

for image_path in images:
    results = model(str(image_path))

    for result in results:
        output_path = output_folder / image_path.name
        result.save(filename=str(output_path))

# ----------------------------------    
# Resultados

image_counts = {}

image_name = ""
total_confidence = 0

# print("\nDetecções:")

for detection in detections:

    if image_name != detection["image"]:
        image_name = detection["image"]
        # print(f"\n{image_name}")

        image_counts[image_name] = 0

    image_counts[image_name] += 1

    category = detection["category"]
    vehicle_counts[category] += 1

    total_confidence += detection["confidence"]

    # print(detection)



# ----------------------------------
# # Totais

# print("\nTotal de veículos: ", len(detections))

# print("\nTotal de veículos por categoria:")
# for category, count in vehicle_counts.items():
#     print(f"{category}: {count}")

# print(image_counts)
# print(f"\nImagem com mais detecções: {max(image_counts, key=image_counts.get)}")

# # Media de confiança
# if len(detections) > 0:
#     average_confidence = total_confidence / len(detections)
# else:
#     average_confidence = 0

# print(f"\nConfiança média: {average_confidence:.2%}")



# ----------------------------------
# Criação do DataFrame

df = pd.DataFrame(detections)

print(df)

print("\nPrimeiras linhas:")
print(df.head())

print("\nInformações:")
print(df.info())

print("\nDimensões:")
print(df.shape)

# Salvando o DataFrame como CSV
output_folder = Path("results")
output_folder.mkdir(exist_ok=True)
df.to_csv(output_folder / "detections.csv", index=False)


print("\nTotal de veículos: ", len(df))

print(f"\nMédia de confiança: {df['confidence'].mean():.2%}")

print(f"\nTotal de ônibus: \n{df[df['category'] == 'bus']}")

print(f"\nVeículos por categoria: \n{df['category'].value_counts()}")


vehicles_per_images = df.groupby('image')['category'].count()

image_more_vh = vehicles_per_images.idxmax()

qtt_vehicles = vehicles_per_images.max()

print(f"\nImagem com mais veículos: {image_more_vh} com {qtt_vehicles}")

# ----------------------------------
# Estatística Descritiva

print("\nEstatística Descritiva:")

print(vehicles_per_images.describe())

print(f"Média de veículos por imagem: {vehicles_per_images.mean()}")

print("\nMediana: ")
print(vehicles_per_images.median())

print("\nModa: ")
print(vehicles_per_images.mode())

print("\nDesvio padrão: ")
print(vehicles_per_images.std())

print("\nMínimo: ")
print(vehicles_per_images.min())
print("\nMáximo: ")
print(vehicles_per_images.max())

print("\nQ0 (mínimo): ")
print(vehicles_per_images.quantile(0))
print("\nQ1: ")
print(vehicles_per_images.quantile(0.25))
print("\nQ2: ")
print(vehicles_per_images.quantile(0.50))
print("\nQ3: ")
print(vehicles_per_images.quantile(0.75))
print("\nQ4 (máximo): ")
print(vehicles_per_images.quantile(1))
