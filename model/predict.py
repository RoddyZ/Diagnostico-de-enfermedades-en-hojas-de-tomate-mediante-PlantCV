import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from model import baseLineModel  # Importa la arquitectura del modelo

def predict_image(image_path, model=None, weights_path=None, class_names=None, device=None):
    """
    Carga una imagen desde una ruta, la preprocesa y predice su clase con el modelo entrenado.
    Si no se proporciona un modelo, se carga desde los pesos indicados.

    Parámetros:
    - image_path (str): Ruta de la imagen.
    - model (torch.nn.Module, opcional): Modelo entrenado. Si no se proporciona, se cargará desde los pesos.
    - weights_path (str, opcional): Ruta del archivo .pth con los pesos guardados. Requerido si `model` es None.
    - class_names (list): Lista de nombres de clases.
    - device (torch.device, opcional): CPU o GPU (se detecta automáticamente si no se especifica).

    Retorna:
    - str: Nombre de la clase predicha.
    - float: Probabilidad de la clase predicha (entre 0 y 1).
    """

    # Validaciones
    if model is None and weights_path is None:
        raise ValueError("Debes proporcionar un modelo entrenado o la ruta de los pesos.")

    if class_names is None:
        raise ValueError("Se requiere la lista de nombres de clases para realizar la predicción.")

    # Definir el dispositivo
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Si no se proporciona el modelo, cargarlo desde los pesos
    if model is None:
        num_classes = len(class_names)  # Determinar el número de clases
        model = baseLineModel(num_classes)  # Inicializar el modelo
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()  # Poner en modo evaluación
        print(f"Modelo cargado desde {weights_path}")

    # Transformaciones de la imagen
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ajustar al tamaño de entrada del modelo
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalización para imágenes en escala de grises
    ])

    # Cargar la imagen y convertirla a escala de grises
    image = Image.open(image_path).convert("L")

    # Aplicar transformaciones
    image = transform(image).unsqueeze(0)  # Añadir dimensión de batch

    # Enviar la imagen y el modelo al dispositivo
    image = image.to(device)
    model.eval()  # Asegurar que el modelo está en modo evaluación

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Aplicar softmax para obtener probabilidades
        predicted_prob, predicted_idx = torch.max(probabilities, 1)  # Obtener la clase con mayor probabilidad
        predicted_class = class_names[predicted_idx.item()]  # Obtener el nombre de la clase
        predicted_prob = predicted_prob.item()  # Convertir a número flotante

    return predicted_class, predicted_prob
