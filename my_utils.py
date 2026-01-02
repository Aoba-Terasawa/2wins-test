import os
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def fix_seed(seed=42):
    """
    再現性を確保するためにシードを固定する関数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed fixed to {seed}")

def calculate_confusion_matrix(model, dataloader, device, class_names=None, threshold=0.5):
    """
    混同行列を計算し、表示する関数 
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            good_probs = probs[:, 1]
            
            pred_indices = (good_probs >= threshold).long()
            
            all_preds.extend(pred_indices.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n--- Confusion Matrix (Threshold={threshold}) ---")
    print(cm)
    
    if class_names:
        print("\n--- Classification Report ---")
        print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\n--- Miss List ---")
    for i, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
        if true_label != pred_label:
            print(f"Index: {i}")
            print(f"True Label: {class_names[true_label]} ({true_label})")
            print(f"Predicted: {class_names[pred_label]} ({pred_label})")
            print(f"Path: {dataloader.dataset.images_paths[i]}")
            print()
    
    return cm

def plot_confusion_matrix(cm, class_names):
    """
    混同行列をヒートマップとしてプロットする関数 
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()