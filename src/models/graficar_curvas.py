import matplotlib.pyplot as plt
import numpy as np

def graficarCurvas(history):
    
    # Extraer datos del entrenamiento
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Calcular precisión total como promedio ponderado
    # (Si ambas salidas tienen igual peso, se usa el promedio simple)
    total_accuracy = np.mean([
        history.history['disease_output_accuracy'], 
        history.history['plant_output_accuracy']
    ], axis=0)
    
    val_total_accuracy = np.mean([
        history.history['val_disease_output_accuracy'], 
        history.history['val_plant_output_accuracy']
    ], axis=0)
    
    # Configurar gráficos
    plt.figure(figsize=(15, 12))
    
    # Pérdida total
    plt.subplot(4, 2, 1)
    plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.legend()
    
    # Precisión total
    plt.subplot(4, 2, 2)
    plt.plot(epochs, total_accuracy, 'b', label='Training Total Accuracy')
    plt.plot(epochs, val_total_accuracy, 'r', label='Validation Total Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Total Accuracy')
    plt.legend()
    
    # Pérdida para Disease Output
    plt.subplot(4, 2, 3)
    plt.plot(epochs, history.history['disease_output_loss'], 'b', label='Training Disease Loss')
    plt.plot(epochs, history.history['val_disease_output_loss'], 'r', label='Validation Disease Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Disease Loss')
    plt.legend()
    
    # Precisión para Disease Output
    plt.subplot(4, 2, 4)
    plt.plot(epochs, history.history['disease_output_accuracy'], 'b', label='Training Disease Accuracy')
    plt.plot(epochs, history.history['val_disease_output_accuracy'], 'r', label='Validation Disease Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Disease Accuracy')
    plt.legend()
    
    # Pérdida para Plant Output
    plt.subplot(4, 2, 5)
    plt.plot(epochs, history.history['plant_output_loss'], 'b', label='Training Plant Loss')
    plt.plot(epochs, history.history['val_plant_output_loss'], 'r', label='Validation Plant Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Plant Loss')
    plt.legend()
    
    # Precisión para Plant Output
    plt.subplot(4, 2, 6)
    plt.plot(epochs, history.history['plant_output_accuracy'], 'b', label='Training Plant Accuracy')
    plt.plot(epochs, history.history['val_plant_output_accuracy'], 'r', label='Validation Plant Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Plant Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    