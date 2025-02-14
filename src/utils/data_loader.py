import os
import cv2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DatasetFolder
import torch

def plantcv_preprocess(image_path):
    image_path = os.path.abspath(image_path)  # Convertir a ruta absoluta
    # Intentar cargar la imagen
    image = cv2.imread(image_path)

    # Si no se pudo cargar, intentar con otra opción
    if image is None:
        print(f"⚠️ Advertencia: No se pudo cargar la imagen {image_path}. Intentando con IMREAD_UNCHANGED...")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Si sigue siendo None, ignoramos la imagen
    if image is None:
        print(f"❌ Error: Imagen {image_path} no se pudo cargar. Se omitirá en el procesamiento.")
        return None  # Retorna None para que el DataLoader lo maneje

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convertir a escala de grises usando OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Aplicar umbralado
    _, thresh_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    # Redimensionar y normalizar
    image_resized = cv2.resize(thresh_img, (256, 256))
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalización
    
    return image_tensor

def custom_loader(path):
    return plantcv_preprocess(path)

def get_data_loaders(data_dir, batch_size=32, valid_split=0.1):
    train_dataset = DatasetFolder(root=os.path.join(data_dir, 'train'), loader=custom_loader, extensions=('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG'))
    test_dataset = DatasetFolder(root=os.path.join(data_dir, 'valid'), loader=custom_loader, extensions=('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG'))  # Ahora es test
    
    # Separar 10% de entrenamiento para validación
    total_train = len(train_dataset)
    valid_size = int(valid_split * total_train)
    train_size = total_train - valid_size
    
    train_data, valid_data = random_split(train_dataset, [train_size, valid_size])

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Antes era valid_loader

    return train_loader, valid_loader, test_loader, train_dataset.classes

