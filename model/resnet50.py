# This files use to revised default ResNet-50
# Since SimSiam's intput dimension of Projection must be 2048

import torch.nn as nn
import torchvision.models as models

class CustomizedResnet50(nn.Module):
    def __init__(self, num_classes=2048):
        super().__init__()

        self.resnet = models.resnet50()

        # modified last fc layer
        in_features = self.resnet.fc.in_features # 2048
        self.resnet.fc = nn.Linear(in_features, num_classes) # [B, 2048] -> [B, 2048]
    
    def forward(self, x):
        return self.resnet(x)

class CustomizedEfficientNetB4(nn.Module):
    def __init__(self, num_classes=2048):
        super().__init__()

        self.efficientnetb4 = models.efficientnet_b4()

        # modified last fc layer
        in_features = self.efficientnetb4.classifier[1].in_features # 1792
        self.efficientnetb4.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.efficientnetb4(x)
        
        