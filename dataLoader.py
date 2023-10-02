import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import os

import customDataset
from customDataset import TrafficDataset, Information


BATCH_SIZE = 12
# Result from meanAndStd.py: 
# (mean, std) = (tensor([0.4801, 0.4757, 0.4846]), tensor([0.1815, 0.1702, 0.1645]))
# mean = [0.4801, 0.4757, 0.4846]
# std = [0.1815, 0.1702, 0.1645]

mean= [0.47971445, 0.4753267,  0.48419115]
std =[0.22706066, 0.21074826, 0.20439357]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

train_dataset = TrafficDataset(annotations_file=customDataset.csv_name, img_dir=customDataset.training_dataset_path, information=Information.VELOCITY, transform=train_transforms)
train_loader = DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Uncommnet for testing dataloader
# batch = next(iter(train_loader))
# images, labels = batch
# print(images.shape)
# grid = torchvision.utils.make_grid(images, nrow=3)
# plt.figure(figsize=(11, 11))
# plt.imshow(np.transpose(grid, (1, 2 , 0)))
# plt.show()
# print('labels: ', labels)

