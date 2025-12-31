import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ProductDataset(Dataset):
    def __init__(self, images_paths, labels, transform=None):
        self.images_paths = images_paths
        self.transform = transform
        self.labels = labels
    
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data(data_dir):
    all_images_paths = []
    all_labels = []
    class_map = {"bad": 0, "good": 1}

    for class_name, label in class_map.items():
        pattern = os.path.join(data_dir, class_name, "*")
        files = glob.glob(pattern)
        all_images_paths.extend(files)
        all_labels.extend([label] * len(files))

    return all_images_paths, all_labels, list(class_map.keys())

if __name__ == '__main__':
    datadir = r"C:\Users\user\2wins-test\dataset"
    if not os.path.exists(datadir):
        datadir = "dataset"

    all_images_paths, all_labels, class_names = prepare_data(datadir)
    if not all_images_paths:
        print("エラー: 画像が見つかりませんでした。")
        exit()

    train_images_paths, temp_images_paths, train_labels, temp_labels = train_test_split(
        all_images_paths, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    val_images_paths, test_images_paths, val_labels, test_labels = train_test_split(
        temp_images_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"データ数 - Train: {len(train_images_paths)}, Val: {len(val_images_paths)}, Test: {len(test_images_paths)}")