import torch
import torch.nn as nn
import torchvision.models as models

class CatDogClassifier(nn.Module):
    """
    Cat vs Dog Classifier using Transfer Learning.
    Based on the VGG16 / Pretrained CNN transfer learning workflow shown in the notebook.
    
    Structure:
    - Pretrained Convolutional Base (frozen feature extractor)
    - Custom Classification Head:
        - Flatten / Adaptive Pooling
        - Dense / Linear(256) + ReLU
        - Dropout(0.5)
        - Dense / Linear(2) for Cat & Dog logits
    """
    def __init__(self, backbone="mobilenet_v2", num_classes=2, freeze_base=True, pretrained=True):
        super(CatDogClassifier, self).__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = num_classes

        if self.backbone_name == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            base = models.vgg16(weights=weights)
            
            # Freeze convolutional base feature extractor
            if freeze_base:
                for param in base.features.parameters():
                    param.requires_grad = False
            
            self.features = base.features
            self.avgpool = base.avgpool
            
            # Custom Classifier Head matching the notebook image:
            # Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(num_classes)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )

        elif self.backbone_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v2(weights=weights)
            
            if freeze_base:
                for param in base.features.parameters():
                    param.requires_grad = False
            
            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Head matching notebook: Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(num_classes)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base.last_channel, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )
        else:
            # Simple custom CNN architecture (VGG-style)
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # 112x112

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # 56x56

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), # 28x28

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 14x14
            )
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256 * 7 * 7, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

def build_model(backbone="mobilenet_v2", num_classes=2, pretrained=True, freeze_base=True):
    model = CatDogClassifier(backbone=backbone, num_classes=num_classes, freeze_base=freeze_base, pretrained=pretrained)
    return model
