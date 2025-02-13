import os
import cv2
import plantcv as pcv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import numpy as np
import torch

def plantcv_preprocess(image_path):
    image = cv2.imread(image_path)
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

def get_data_loaders(data_dir, batch_size=32):
    train_dataset = DatasetFolder(root=os.path.join(data_dir, 'train'), loader=custom_loader, extensions=('jpg', 'png', 'jpeg','JPG', 'PNG', 'JPEG'))
    valid_dataset = DatasetFolder(root=os.path.join(data_dir, 'valid'), loader=custom_loader, extensions=('jpg', 'png', 'jpeg','JPG', 'PNG', 'JPEG'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, train_dataset.classes
