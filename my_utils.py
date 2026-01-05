import os
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

def find_best_threshold(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            good_probs = probs[:, 1]
            
            all_preds.extend(good_probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

    tmp = np.where(fpr <= 0)[0]
    idx = tmp[-1]
    best_threshold = thresholds[idx]
    
    return best_threshold


    
    

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

def save_gradcam_image(model, target_layer, image_path, transform, device, save_path):
    model.eval()

    rgb_img = Image.open(image_path).convert('RGB')
    rgb_img = rgb_img.resize((448, 448))
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    rgb_img_float = np.array(rgb_img).astype(np.float32) / 255.0

    visualization_array = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    
    visualization_image = Image.fromarray(visualization_array)
    visualization_image.save(save_path)
    
    print(f"Grad-CAM result saved (PIL) to: {save_path}")
    
    