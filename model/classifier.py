import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, model, n_classes, phase=None):
        """
        Parameter
        ------------
        model : pre-trained model
        n_classes : output classes
        phase : only "fine-tune" or "Linear"
        """
        super().__init__()
        
        in_features = model.fc.in_features

        self.model = model
        self.model.fc = nn.Linear(in_features, n_classes)
        
        if phase == "Linear":
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    