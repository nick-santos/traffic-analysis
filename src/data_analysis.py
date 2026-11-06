import pandas as pd


def create_dataframe(detections):
    return pd.DataFrame(detections)


def save_dataframe_csv(df, results_folder):
    df.to_csv(results_folder / "detections.csv", index=False)


def check_dataframe_info(df):
    print("\nPrimeiras linhas:")
    print(df.head())

    print("\nInformações:")
    df.info()

    print("\nDimensões:")
    print(df.shape)


def analyze_vehicles(df):
    if df.empty:
        print("Nenhum veículo foi detectado.")
        return None, None

    vehicles_per_image = df.groupby('image')['category'].count()
    vehicles_per_category = df["category"].value_counts()

    print("\n--- Análise dos veículos ---")

    print(f"Total de veículos: {len(df)}")

    # print(f"\nDetecções de ônibus: \n{df[df['category'] == 'bus']}")

    print(f"Média de confiança: {df['confidence'].mean():.2%}")

    image_with_most_vehicles = vehicles_per_image.idxmax()
    vehicle_count = vehicles_per_image.max()
    print(f"\nImagem com mais veículos: {image_with_most_vehicles} com {vehicle_count} veículos")

    print("\nVeículos por categoria:")
    print(vehicles_per_category)

    print("\nVeículos por imagem:")
    print(vehicles_per_image)

    return vehicles_per_image, vehicles_per_category


def descriptive_statistics(vehicles_per_image):
    print("\n--- Estatística descritiva ---")
    print(vehicles_per_image.describe())

    print("\nMédia de veículos por imagem:")
    print(vehicles_per_image.mean())

    print("\nMediana: ")
    print(vehicles_per_image.median())

    print("\nModa: ")
    print(vehicles_per_image.mode())

    print("\nDesvio padrão: ")
    print(vehicles_per_image.std())

    print("\nMínimo: ")
    print(vehicles_per_image.min())
    print("\nMáximo: ")
    print(vehicles_per_image.max())

    print("\nQ0 (mínimo): ")
    print(vehicles_per_image.quantile(0))
    print("\nQ1: ")
    print(vehicles_per_image.quantile(0.25))
    print("\nQ2: ")
    print(vehicles_per_image.quantile(0.50))
    print("\nQ3: ")
    print(vehicles_per_image.quantile(0.75))
    print("\nQ4 (máximo): ")
    print(vehicles_per_image.quantile(1))


def analyze_detections_raw(detections):
    vehicles_per_image = {}
    image_name = ""
    total_confidence = 0

    vehicles_per_category = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    print("\n--- Detecções ---")

    for detection in detections:

        if image_name != detection["image"]:
            image_name = detection["image"]
            print(f"\n{image_name}")

            vehicles_per_image[image_name] = 0

        vehicles_per_image[image_name] += 1

        category = detection["category"]
        vehicles_per_category[category] += 1

        total_confidence += detection["confidence"]

        print(detection)

    # Análise sobre detections

    print("\n--- Análise inicial ---")

    print("\nTotal de veículos: ", len(detections))

    print("\nTotal de veículos por categoria:")
    for category, count in vehicles_per_category.items():
        print(f"{category}: {count}")

    if vehicles_per_image:
        print(f"\nImagem com mais detecções: {max(vehicles_per_image, key=vehicles_per_image.get)}")
    else:
        print("\nNenhum veículo foi detectado.")

    if len(detections) > 0:
        average_confidence = total_confidence / len(detections)
    else:
        average_confidence = 0

    print(f"\nConfiança média: {average_confidence:.2%}")
