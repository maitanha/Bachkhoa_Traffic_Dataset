import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torchvision.models as models
from helper.loader import get_data_loader

model = models.resnet18()
model_name = "./pretrained/resnet18-f37072fd.pth"
model.load_state_dict(torch.load(model_name))

# Modify the last layer for regression (1 output)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)  # 1 output neuron for regression

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class Model:
    def __init__(
        self, 
        data_directory, 
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        batch_size=32, 
        transform=transform
        ):
        """
        Args:
            data_directory (_type_): _description_
            model (_type_, optional): _description_. Defaults to ResNet18.
            criterion (_type_, optional): _description_. Defaults to MSE.
            optimizer (_type_, optional): _description_. Defaults to Adam.
            batch_size (int, optional): _description_. Defaults to 32.
            transform (_type_, optional): _description_. Defaults to None.
        """
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.transform = transform
        
    def train(
        self, 
        epochs = 10, 
        store_model = True, 
        model_name = "resnet_traffic_model.pth"
        ):
        """
        Train and save model as pth file
        Args:
            epochs (int, optional): _description_. Defaults to 10.
            store_model (bool, optional): _description_. Defaults to True.
            model_name (str, optional): _description_. Defaults to "resnet_traffic_model.pth" weights by resnet18.
        """
        
        train_loader, _ = self.get_data_loader()

        for epoch in range(epochs):
            print(f"Running epoch {epoch+1}:")
            for images, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets.view(-1, 1).float())  # Targets should be a float
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

        if store_model:
            torch.save(model.state_dict(), "./pretrained/" + model_name)
            
    def validate(
        self, 
        model_name="resnet_traffic_model.pth" 
        ):
        """
        Args:
            model_name (str, optional): _description_. Defaults to "resnet_traffic_model.pth".

        Returns:
           expectations (list): _description_
           predictions (list): _description_
        """
        
        validation_model = self.model
        validation_model.load_state_dict(torch.load("./pretrained/" + model_name))
        
        _ , validation_loader = self.get_data_loader()
        
        predictions = []
        expectations = []
        
        with torch.no_grad():
            for images, labels in validation_loader:
                outputs = validation_model(images)
                expectations.append(labels)
                predictions.extend(outputs.cpu().numpy())
        
        # Switch validation_model to train mode
        validation_model.train()
        
        # Convert to numpy array
        expectations = torch.cat(expectations).cpu().numpy()
        predictions = np.array(predictions).reshape(-1)
        
        return expectations, predictions
        
        
    def get_data_loader(self):
        train_loader, validation_loader = get_data_loader(
            data_dir=self.data_directory,
            batch_size=self.batch_size, 
            train_transform=self.transform, 
            val_transform=self.transform, 
            shuffle=True
        )
        return train_loader, validation_loader
    