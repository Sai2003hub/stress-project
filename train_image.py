import sys
import os

# Add the project root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from dataset import create_data_loader  # Ensure this import is correct
from multiprocessing import freeze_support

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss increased ({self.best_loss:.6f} --> {val_loss:.6f}). {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

# Data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image_data(csv_path):
    data = pd.read_csv(csv_path)
    # Convert the 'pixels' column from strings to image arrays
    data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48, 1))
    # Normalize pixel values to the range [0, 1]
    data['pixels'] = data['pixels'] / 255.0
    # Split data into training, validation, and test sets based on the 'Usage' column
    train_data = data[data['Usage'] == 'Training']
    val_data = data[data['Usage'] == 'PublicTest']
    test_data = data[data['Usage'] == 'PrivateTest']
    # Save preprocessed data
    np.save('C:/Users/Adithya/Downloads/idk1/data/train_data.npy', train_data)
    np.save('C:/Users/Adithya/Downloads/idk1/data/val_data.npy', val_data)
    np.save('C:/Users/Adithya/Downloads/idk1/data/test_data.npy', test_data)
    return train_data, val_data, test_data

def train_image_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    accumulation_steps = 4  # For gradient accumulation
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/total}, Accuracy: {100 * correct/total}%, Val Loss: {val_loss}')
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # Learning rate scheduler
        scheduler.step(val_loss)

    # Load the best model
    model.load_state_dict(early_stopping.best_model)
    return model

if __name__ == '__main__':
    # Ensure to include freeze_support() if needed
    freeze_support()

    # Check if GPU is available
    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used for training.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Training will be done on CPU.")
        device = torch.device("cpu")

    # Verify CUDA
    print("CUDA Available: ", torch.cuda.is_available())
    print("Number of GPUs: ", torch.cuda.device_count())
    print("GPU Name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

    # Load preprocessed data
    csv_path = 'C:/Users/Adithya/Downloads/idk1/data/fer2013.csv'
    train_data, val_data, test_data = preprocess_image_data(csv_path)
    train_loader = create_data_loader(train_data, batch_size=16, shuffle=True, transforms=train_transforms)
    val_loader = create_data_loader(val_data, batch_size=16, shuffle=False, transforms=val_transforms)

    # Model and Training Setup
    image_model = models.resnet18(pretrained=True)
    num_ftrs = image_model.fc.in_features
    image_model.fc = torch.nn.Linear(num_ftrs, 7)  # Assuming 7 emotion classes
    image_model = image_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(image_model.parameters(), lr=2e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Train the model with early stopping
    trained_model = train_image_model(image_model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, patience=5)

    # Save the best model
    save_dir = 'C:/Users/Adithya/Downloads/idk1/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(save_dir, 'best_image_model.pth'))
    print("Best model saved successfully.")
