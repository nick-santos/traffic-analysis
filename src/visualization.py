import matplotlib.pyplot as plt


def plot_vehicles_per_category(vehicles_per_category, results_graphs_folder):
    plt.figure(figsize=(8, 5))
    
    vehicles_per_category.plot(kind='bar')

    plt.title('Quantidade de veículos por categoria')
    plt.xlabel('Categoria')
    plt.ylabel('Quantidade de veículos')

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(results_graphs_folder/'vehicles_per_category.png')
    plt.show()


def plot_vehicles_per_image(vehicles_per_image, results_graphs_folder):
    plt.figure(figsize=(8, 5))

    vehicles_per_image.plot(kind='bar')

    plt.title('Quantidade de veículos por imagem')
    plt.xlabel('Imagem')
    plt.ylabel('Quantidade de veículos')

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(results_graphs_folder/'vehicles_per_image.png')
    plt.show()

