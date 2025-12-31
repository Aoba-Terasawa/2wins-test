import os
import glob
import time
import copy
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from my_utils import fix_seed, calculate_confusion_matrix

DATA_DIR = r"C:\Users\user\2wins-test\dataset"
TEST_SIZE = 0.3
VAL_SIZE = 0.5  # train : val : test = 0.7 : 0.15 : 0.15
INPUT_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
SEED = 42
SAVE_NAME = "resnet18_base.pth"

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

def create_model(num_classes):
    print("ResNet18モデルをロード中...")
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    return model_ft

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    fix_seed(SEED)

    datadir = DATA_DIR
    if not os.path.exists(datadir):
        datadir = "dataset"

    all_images_paths, all_labels, class_names = prepare_data(datadir)
    if not all_images_paths:
        print("エラー: 画像が見つかりませんでした。")
        exit()

    train_images_paths, temp_images_paths, train_labels, temp_labels = train_test_split(
        all_images_paths, all_labels, test_size=TEST_SIZE, random_state=SEED, stratify=all_labels
    )
    val_images_paths, test_images_paths, val_labels, test_labels = train_test_split(
        temp_images_paths, temp_labels, test_size=VAL_SIZE, random_state=SEED, stratify=temp_labels
    )

    print(f"データ数 - Train: {len(train_images_paths)}, Val: {len(val_images_paths)}, Test: {len(test_images_paths)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    input_size = INPUT_SIZE

    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ProductDataset(train_images_paths, train_labels, transform=transform_train)
    val_dataset = ProductDataset(val_images_paths, val_labels, transform=transform_val)

    batch_size = BATCH_SIZE
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    model_ft = create_model(len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    num_epochs = NUM_EPOCHS

    model_ft = train_model(model_ft, criterion, optimizer_ft, dataloaders, dataset_sizes, device, num_epochs=num_epochs)

    torch.save(model_ft.state_dict(), SAVE_NAME)
    print(f"モデルを保存しました ({SAVE_NAME})。")

    # 混同行列の計算
    calculate_confusion_matrix(model_ft, dataloaders['val'], device, class_names)