from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Temporary
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.count_label = {}
        self.file_list = self._load_file_list()

    def _load_file_list(self):
        file_list = []
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            label = self._extract_label_from_filename(file_name)

            file_list.append((file_path, label))
        return file_list


    def _extract_label_from_filename(self, filename):
        # Extract the label from the filename, number -1 is to get the last part of the filename
        splitted_name = filename.split('_')[-1]
        label_without_extension = splitted_name.split('.')[0]
        density_label = float(label_without_extension)
        
        return density_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_velocity_distribution(self):
        for file_name in os.listdir(self.data_dir):
            label = self._extract_label_from_filename(file_name)

            # Initialize the count_label entry for the current label if not present
            if label not in self.count_label:
                self.count_label[label] = 0

            self.count_label[label] += 1
        return self.count_label
    
    
def get_data_loader(data_dir, batch_size, train_transform, val_transform, num_workers=0, shuffle=True):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = CustomDataset(train_dir, transform=train_transform)
    val_dataset = CustomDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, val_loader