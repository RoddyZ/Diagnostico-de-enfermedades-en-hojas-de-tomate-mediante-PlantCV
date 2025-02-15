import torch
import torch.nn as nn
import torch.optim as optim

def train_model(
    model, train_loader, valid_loader, num_epochs=10, learning_rate=0.001,
    criterion=None, optimizer=None, scheduler=None, device=None
):
    # Definir el dispositivo (CPU/GPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Entrenando en: {device}")  
    model.to(device)
    
    # Usar función de pérdida personalizada o CrossEntropy por defecto
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Usar optimizador personalizado o Adam por defecto
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Guardar métricas
    history = {
        "train_loss": [], "train_acc": [],
        "valid_loss": [], "valid_acc": []
    }

    for epoch in range(num_epochs):
        # --- Entrenamiento ---
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # --- Validación ---
        model.eval()
        running_valid_loss, correct_valid, total_valid = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_valid += labels.size(0)
                correct_valid += (predicted == labels).sum().item()

        avg_valid_loss = running_valid_loss / len(valid_loader)
        valid_accuracy = 100 * correct_valid / total_valid
        
        # Guardar métricas
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)
        history["valid_loss"].append(avg_valid_loss)
        history["valid_acc"].append(valid_accuracy)

        # Ajustar learning rate si se usa scheduler
        if scheduler:
            scheduler.step(avg_valid_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}% - "
              f"Valid Loss: {avg_valid_loss:.4f} - Valid Accuracy: {valid_accuracy:.2f}%")

    return model, history
