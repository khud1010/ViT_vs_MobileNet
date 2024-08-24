import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights, ViT_B_16_Weights
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 256
learning_rate = 0.001
num_classes = 2  # Binary classification
image_size = 224  # Adjusted input size for ViT

# Data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Ensure images are the correct size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths to data
data_path = "./binary_data"
train_dir = os.path.join(data_path, 'train_images')
test_dir = os.path.join(data_path, 'test_images')

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define MobileNet model
mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
mobilenet = mobilenet.to(device)

# Define Vision Transformer model
vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
vit.heads.head = nn.Linear(vit.heads.head.in_features, num_classes)
vit = vit.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
mobilenet_optimizer = optim.Adam(mobilenet.parameters(), lr=learning_rate)
vit_optimizer = optim.Adam(vit.parameters(), lr=learning_rate)

# Training function with time tracking
def train_model(model, optimizer, num_epochs, model_name):
    model.train()
    train_losses = []
    epoch_times = []  # List to store the time taken for each epoch

    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time = time.time()  # Start time for the epoch

        print(f"Starting epoch {epoch + 1}/{num_epochs} for {model_name}")
        for images, labels in tqdm(train_loader, total=len(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Calculate and store the time taken for the epoch
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} seconds')
    
    return train_losses, epoch_times

# Evaluation function
def evaluate_model(model, model_name):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluating batch [{batch_idx + 1}/{len(test_loader)}] for {model_name}")

    accuracy = 100 * correct / total
    return accuracy, all_labels, all_predictions

# Plotting function updated to include training time
def plot_metrics(train_losses, epoch_times, accuracy, labels, predictions, model_name):
    sns.set(style="whitegrid")

    # Plot training loss over epochs
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training Loss')
    plt.legend()

    # Plot training time over epochs
    plt.subplot(1, 3, 2)
    plt.plot(epoch_times, label='Training Time')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.title(f'{model_name} - Training Time per Epoch')
    plt.legend()

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(labels, predictions)
    plt.subplot(1, 3, 3)
    ConfusionMatrixDisplay(cm, display_labels=train_dataset.classes).plot(cmap='Blues', ax=plt.gca())
    plt.title(f'{model_name} - Confusion Matrix')
    
    plt.show()

    print(f'{model_name} Accuracy: {accuracy:.2f}%')


# Train and evaluate MobileNet
print("Training MobileNet...")
mobilenet_train_losses, mobilenet_epoch_times = train_model(mobilenet, mobilenet_optimizer, num_epochs, "MobileNet")
mobilenet_accuracy, mobilenet_labels, mobilenet_predictions = evaluate_model(mobilenet, "MobileNet")
plot_metrics(mobilenet_train_losses, mobilenet_epoch_times, mobilenet_accuracy, mobilenet_labels, mobilenet_predictions, "MobileNet")


# Train and evaluate Vision Transformer
print("Training Vision Transformer...")
vit_train_losses, vit_epoch_times = train_model(vit, vit_optimizer, num_epochs, "Vision Transformer")
vit_accuracy, vit_labels, vit_predictions = evaluate_model(vit, "Vision Transformer")
plot_metrics(vit_train_losses, vit_epoch_times, vit_accuracy, vit_labels, vit_predictions, "Vision Transformer")
