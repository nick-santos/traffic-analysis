from ultralytics import YOLO
from pathlib import Path

from src.detection import (
    detect_vehicles, 
    save_detection_images 
)
from src.data_analysis import (
    create_dataframe, 
    check_dataframe_info, 
    save_dataframe_csv, 
    analyze_detections_raw,
    analyze_vehicles, 
    descriptive_statistics
)
from src.visualization import (
    plot_vehicles_per_category,
    plot_vehicles_per_image
)

# Pastas do projeto
IMAGE_FOLDER = Path("dataset/images")
RESULTS_FOLDER = Path("results")
RESULTS_IMAGES_FOLDER = RESULTS_FOLDER / "detections"
RESULTS_GRAPHS_FOLDER = RESULTS_FOLDER / "graphs"


def main():
    # Criar pastas de resultados
    RESULTS_FOLDER.mkdir(exist_ok=True)
    RESULTS_IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
    RESULTS_GRAPHS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Carregar o modelo
    model = YOLO("yolo11n.pt")

    # Detecção de veículos
    detections = detect_vehicles(model, IMAGE_FOLDER)

    # Salvar imagens com Bounding Boxes
    save_detection_images(model, RESULTS_IMAGES_FOLDER, IMAGE_FOLDER)

    # Análise das detecções (sem Dataframe)
    analyze_detections_raw(detections)

    # Criação do DataFrame
    df = create_dataframe(detections)

    # Checando informações do Dataframe
    check_dataframe_info(df)

    # Salvando DataFrame como CSV
    save_dataframe_csv(df, RESULTS_FOLDER)

    # Análise de dados com Pandas
    if not df.empty:
        vehicles_per_image, vehicles_per_category = analyze_vehicles(df)

        # Estatística Descritiva
        descriptive_statistics(vehicles_per_image)

    else:
        print("Dataframe vazio.")

    # Visualização dos dados
    if not df.empty:
        plot_vehicles_per_category(vehicles_per_category, RESULTS_GRAPHS_FOLDER)
        plot_vehicles_per_image(vehicles_per_image, RESULTS_GRAPHS_FOLDER)


if __name__ == "__main__":
    main()
