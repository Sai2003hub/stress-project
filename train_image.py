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
from dataset import create_data_loader
from multiprocessing import freeze_support
import torch.utils.checkpoint as checkpoint
from PIL import Image

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
    data['pixels'] = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48))
    data['pixels'] = data['pixels'] / 255.0
    data['pixels'] = data['pixels'].apply(lambda x: np.repeat(x[:, :, np.newaxis], 3, axis=2))
    train_data = data[data['Usage'] == 'Training']
    val_data = data[data['Usage'] == 'PublicTest']
    test_data = data[data['Usage'] == 'PrivateTest']
    return train_data, val_data, test_data

def create_data_loader(data, batch_size, shuffle, transforms):
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data, transforms):
            self.data = data
            self.transforms = transforms

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            pixels = self.data.iloc[idx]['pixels']
            label = self.data.iloc[idx]['emotion']
            image = Image.fromarray((pixels * 255).astype('uint8'), 'RGB')
            image = self.transforms(image)
            return image, label

    dataset = CustomDataset(data, transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def checkpointed_forward_function(*inputs):
    def custom_forward(*inputs):
        return image_model(*inputs)
    return checkpoint.checkpoint(custom_forward, *inputs, use_reentrant=False)

def train_image_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    accumulation_steps = 4
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = checkpointed_forward_function(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        scheduler.step(val_loss)

    model.load_state_dict(early_stopping.best_model)
    return model

if __name__ == '__main__':
    freeze_support()

    if torch.cuda.is_available():
        print("CUDA is available. GPU will be used for training.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Training will be done on CPU.")
        device = torch.device("cpu")

    print("CUDA Available: ", torch.cuda.is_available())
    print("Number of GPUs: ", torch.cuda.device_count())
    print("GPU Name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU available")

    csv_path = 'D:/idk1/data/fer2013.csv'
    train_data, val_data, test_data = preprocess_image_data(csv_path)
    train_loader = create_data_loader(train_data, batch_size=16, shuffle=True, transforms=train_transforms)
    val_loader = create_data_loader(val_data, batch_size=16, shuffle=False, transforms=val_transforms)

    image_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = image_model.fc.in_features
    image_model.fc = torch.nn.Linear(num_ftrs, 7)
    image_model = image_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(image_model.parameters(), lr=2e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False)

    trained_model = train_image_model(image_model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, patience=5)

    save_dir = 'D:/idk1/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(save_dir, 'best_image_model.pth'))
    print("Best model saved successfully.")
