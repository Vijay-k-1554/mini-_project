import torch
from models.cnn import CNNBackbone
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt

# ---------- Model ----------
model = CNNBackbone()

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------- Load ONE image ----------
image_folder = "data/demo_images_coco"
image_name = os.listdir(image_folder)[0]

image_path = os.path.join(image_folder, image_name)

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

print("Input shape:", image.shape)

# ---------- Forward ----------
out = model(image)

print("Output shape:", out.shape)

feature_map = out[0, 0].detach().numpy()

plt.imshow(feature_map, cmap='viridis')
plt.title("CNN Feature Map")
plt.colorbar()
plt.show()