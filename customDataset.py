import os
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
import cv2
import csv
import PIL
from enum import Enum

csv_name = 'labeled.csv'
training_dataset_path = './training'

def generate_csv_file(folder_path, csv_file_name):
    file_names = os.listdir(folder_path)
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for file_name in file_names:
            temp = file_name.split('_')
            writer.writerow([file_name] + [str(ord(temp[len(temp) - 3]) - ord('A'))])


class Information(Enum):
    CONDITION = 1
    DENSITY = 2
    VELOCITY = 3

class TrafficDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, information, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.information = information
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = PIL.Image.fromarray(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
        try:
            if(self.information == Information.CONDITION):
                label = torch.tensor(self.img_labels.iloc[idx, 1])
            elif(self.information == Information.DENSITY):
                file_name_elements = self.img_labels.iloc[idx, 0].split('_')
                label = torch.tensor(int(file_name_elements[len(file_name_elements)-2]))
            elif(self.information == Information.VELOCITY):
                file_name_elements = self.img_labels.iloc[idx, 0].split('_')
                label = torch.tensor(int(file_name_elements[len(file_name_elements)-1][:-4])) #slicing for removing ".jpg" or ".png"
            else:
                label = torch.tensor(-1)
        except ValueError as e:
            print(f"ValueError: {e}")
            
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
            
        # image = torch.flatten(image, start_dim=1)
        return (image, label)
    

if __name__ == "__main__":
    generate_csv_file(training_dataset_path, csv_name)
    