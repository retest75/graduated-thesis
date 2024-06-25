# Customized ResNet-50 model
# separate its architecture and modified its final output
# then combine all of these components and became a modified model, named CustomizedResNet50

import torch
from torch import nn
from torchsummary import summary
from torchvision.models import resnet50, ResNet50_Weights

class CustomizedResNet50(nn.Module):
    def __init__(self, weights, n_classes=2):
        super().__init__()

        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        
        self.conv = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(in_features=in_features, out_features=n_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        output = self.fc(x)

        return output
    
if __name__ == "__main__":
    model = CustomizedResNet50(ResNet50_Weights.DEFAULT)
    summary(model, input_size=(3, 224, 224))
    for name, _ in model.named_parameters():
        print(name)



