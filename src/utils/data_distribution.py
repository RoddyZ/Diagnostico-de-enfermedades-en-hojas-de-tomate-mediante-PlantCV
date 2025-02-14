import matplotlib.pyplot as plt
import pandas as pd
import torch

def plot_class_distribution(train_loader, valid_loader, test_loader, class_names):
    """
    Genera un gráfico de barras con la distribución de imágenes por clase en los conjuntos de Train, Validation y Test.
    
    Parámetros:
        train_loader: DataLoader del conjunto de entrenamiento.
        valid_loader: DataLoader del conjunto de validación.
        test_loader: DataLoader del conjunto de prueba.
        class_names: Lista de nombres de clases detectadas.
    """

    # Función para contar imágenes por clase en un DataLoader
    def count_images(loader):
        counts = {class_name: 0 for class_name in class_names}
        for _, labels in loader:
            for label in labels:
                counts[class_names[label]] += 1
        return counts

    # Obtener los conteos de imágenes por clase
    train_counts = count_images(train_loader)
    valid_counts = count_images(valid_loader)
    test_counts = count_images(test_loader)

    # Crear DataFrame con los conteos
    df = pd.DataFrame({'Train': train_counts, 'Validation': valid_counts, 'Test': test_counts})

    # Graficar
    ax = df.plot(kind='bar', figsize=(20, 10), colormap='viridis', alpha=0.85, edgecolor='black')

    plt.title('Distribución de Imágenes por Clase', fontsize=16)
    plt.xlabel('Clase', fontsize=14)
    plt.ylabel('Número de Imágenes', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title="Conjunto", fontsize=12)

    # Agregar los valores numéricos encima de cada barra
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, color='black')

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

def dataset_summary(train_data, valid_data, test_data):
    from collections import Counter

    def count_images(dataset):
        """Cuenta cuántas imágenes hay por clase en un conjunto de datos."""
        class_counts = Counter([dataset.dataset.targets[i] for i in dataset.indices]) if isinstance(dataset, torch.utils.data.Subset) else Counter(dataset.targets)
        total_images = sum(class_counts.values())
        return dict(class_counts), total_images

    # Contar imágenes en cada conjunto
    train_counts, train_total = count_images(train_data)
    valid_counts, valid_total = count_images(valid_data)
    test_counts, test_total = count_images(test_data)

    # Obtener nombres de clases
    class_names = train_data.dataset.classes if isinstance(train_data, torch.utils.data.Subset) else train_data.classes

    # Mostrar resumen
    print("\nResumen de imágenes por clase:\n" + "-"*40)

    print(f"📌 Conjunto de Entrenamiento: {train_total} imágenes en total")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {train_counts.get(i, 0)} imágenes")

    print(f"\n📌 Conjunto de Validación: {valid_total} imágenes en total")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {valid_counts.get(i, 0)} imágenes")

    print(f"\n📌 Conjunto de Pruebas: {test_total} imágenes en total")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {test_counts.get(i, 0)} imágenes")
