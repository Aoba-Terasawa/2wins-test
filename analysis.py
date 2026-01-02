import torch
from torchvision import transforms
from main import create_model 
from my_utils import save_gradcam_image
import os
from PIL import Image

analysis_images = [
    # 成功
    r"C:\Users\user\2wins-test\dataset\bad\05_IMG_2E420008_2025-01-17_15-36-29_000706_GlossRatio_cropped.png",
    r"C:\Users\user\2wins-test\dataset\bad\25_IMG_2E420008_2025-02-10_14-57-09_000106_GlossRatio_cropped.png",

    # 失敗
    r"C:\Users\user\2wins-test\dataset\good\GOOD_IMG_2E420008_2025-01-13_16-16-02_000496_GlossRatio_cropped.png",
    r"C:\Users\user\2wins-test\dataset\bad\25_IMG_2E420008_2025-02-10_14-56-26_000099_GlossRatio_cropped.png",
    r"C:\Users\user\2wins-test\dataset\bad\11_IMG_2E420008_2025-01-31_11-03-32_000004_GlossRatio_cropped.png",
    r"C:\Users\user\2wins-test\dataset\bad\11_IMG_2E420008_2025-01-29_15-19-35_000060_GlossRatio_cropped.png",
    r"C:\Users\user\2wins-test\dataset\bad\11_IMG_2E420008_2025-01-29_15-07-18_000021_GlossRatio_cropped.png"
]

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_classes = 2
    model = create_model(num_classes)
    model.load_state_dict(torch.load('resnet18_threshold.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    target_layer = model.layer4[-1]

    input_size = 448
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists("analysis_results"):
        os.makedirs("analysis_results")

    for i, img_path in enumerate(analysis_images):
        if os.path.exists(img_path):
            save_path = f"analysis_results/gradcam_{i}.png"
            save_gradcam_image(model, target_layer, img_path, transform, device, save_path)

            img = Image.open(img_path)
            img_resized = img.resize((448, 448))
            save_path = f"analysis_results/resized_{i}.png"
            img_resized.save(save_path)

        else:
            print(f"File not found: {img_path}")