import torch
import torch.nn as nn
import torchvision.models as models

# =========================
# 3. 模型
# =========================
class SimpleCNN(nn.Module):
    """一个适合 CIFAR-10 入门练习的简单卷积神经网络。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # [B, 32, 32, 32]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # [B, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 16, 16]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # [B, 64, 8, 8]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [B, 128, 8, 8]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # [B, 128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(num_classes: int = 10):
    # 1. 加载 ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # 用预训练

    # 2. 修改最后一层
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
