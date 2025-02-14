import os
import cv2
import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

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

def get_data_loaders(data_dir, batch_size=32):
    train_dataset = DatasetFolder(root=os.path.join(data_dir, 'train'), loader=custom_loader, extensions=('jpg', 'png', 'jpeg'))
    valid_dataset = DatasetFolder(root=os.path.join(data_dir, 'valid'), loader=custom_loader, extensions=('jpg', 'png', 'jpeg'))
    test_dataset = DatasetFolder(root=os.path.join(data_dir, 'test'), loader=custom_loader, extensions=('jpg', 'png', 'jpeg'))  

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  

    return train_loader, valid_loader, test_loader, train_dataset.classes
