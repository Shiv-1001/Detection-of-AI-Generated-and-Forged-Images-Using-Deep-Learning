import torch
from PIL import Image
from torchvision import transforms, datasets

# CHANGE: path to your trained model
MODEL_PATH = r"C:\dev\forgery-project\model.pth"
# CHANGE: the folder you trained on (the one containing REAL/FAKE subfolders)
TRAIN_DIR = r"C:\dev\forgery-project\data\train"
IMG_PATH = r"C:\Users\SHIV\Downloads\imag.jpeg"

# 1. Class mapping, exactly as ImageFolder saw it during training
print("class_to_idx:", datasets.ImageFolder(TRAIN_DIR).class_to_idx)

# 2. Load the model
# If you saved the whole model with torch.save(model, ...):
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
# If you saved only model.state_dict(), instead do:
#   from your_training_file import YourModelClass
#   model = YourModelClass()
#   model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 3. CHANGE: must match your training transforms exactly
tf = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

img = Image.open(IMG_PATH).convert("RGB")
x = tf(img).unsqueeze(0)

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)

print("probs:", probs)
print("predicted index:", probs.argmax(dim=1).item())