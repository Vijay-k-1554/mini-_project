import torch
from models.cnn import CNNBackbone
from models.vit import VisionTransformer
from models.gnn import GNNModel
from PIL import Image
from torchvision import transforms
import os
import matplotlib.pyplot as plt

# ---------- Models ----------
cnn = CNNBackbone()
vit = VisionTransformer()
gnn = GNNModel(
    in_dim=128,
    hidden_dim=128,
    num_layers=2,
    pool="none"
)

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

# ---------- CNN ----------
cnn_out = cnn(image)
print("CNN Output shape:", cnn_out.shape)

# ---------- Visualize CNN feature ----------
feature_map = cnn_out[0, 0].detach().numpy()

plt.imshow(feature_map, cmap='viridis')
plt.title("CNN Feature Map")
plt.colorbar()
plt.show()

# ---------- ViT ----------
vit_out = vit(cnn_out)
print("ViT Output shape:", vit_out.shape)

# ---------- Inspect tokens ----------
print("First token sample:", vit_out[0, 0, :5])

# ---------- GNN (UPDATED) ----------
gnn_out = gnn(vit_out)
print("GNN Output shape:", gnn_out.shape)

# ---------- Inspect GNN output ----------
print("First GNN node sample:", gnn_out[0, 0, :5])
