from typing import Tuple
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from train import MODEL_PATH, CLASSES, DEVICE
from model import get_model

def get_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


@torch.no_grad()
def predict_image(image_path: str) -> Tuple[str, float]:
    model = get_model(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, dim=1)

    return CLASSES[pred.item()], conf.item()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    label, confidence = predict_image(image_path)
    print(f"Prediction: {label}, confidence: {confidence:.4f}")
