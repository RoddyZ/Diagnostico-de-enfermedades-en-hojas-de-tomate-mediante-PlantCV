import os
import cv2
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, random_split
import torch

def plantcv_preprocess(image_path):
    image_path = os.path.abspath(image_path)  # Convertir a ruta absoluta
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Advertencia: No se pudo cargar la imagen {image_path}. Intentando con IMREAD_UNCHANGED...")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"❌ Error: Imagen {image_path} no se pudo cargar. Se omitirá en el procesamiento.")
        return None  

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    _, thresh_img = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    image_resized = cv2.resize(thresh_img, (256, 256))
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).unsqueeze(0) / 255.0  
    
    return image_tensor

def custom_loader(path):
    return plantcv_preprocess(path)

def get_data_loaders(data_dir, batch_size=32, valid_split=0.1):
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')

    train_dataset = DatasetFolder(root=train_path, loader=custom_loader, extensions=('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG'))
    
    if os.path.exists(test_path):
        print("✅ Carpeta 'test' detectada. Se usará como conjunto de prueba.")
        test_dataset = DatasetFolder(root=test_path, loader=custom_loader, extensions=('jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG'))
        
        # División de validación dentro del conjunto de entrenamiento
        total_train = len(train_dataset)
        valid_size = int(valid_split * total_train)
        train_size = total_train - valid_size

        train_data, valid_data = random_split(train_dataset, [train_size, valid_size])

    else:
        print("⚠️ No se encontró la carpeta 'test'. Se dividirá el conjunto de entrenamiento.")
        total_train = len(train_dataset)
        valid_size = int(valid_split * total_train)
        test_size = valid_size  # Usar mismo tamaño para test si no hay carpeta 'test'
        train_size = total_train - valid_size - test_size

        train_data, valid_data, test_data = random_split(train_dataset, [train_size, valid_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset if os.path.exists(test_path) else test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, train_dataset.classes
