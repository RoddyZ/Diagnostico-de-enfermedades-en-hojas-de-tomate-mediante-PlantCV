import torch
import torch.nn as nn

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Modo evaluación

    correct, total = 0, 0
    running_test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # No necesitamos calcular gradientes
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = running_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    print(f"Test Loss: {avg_test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
    
    return avg_test_loss, test_accuracy
