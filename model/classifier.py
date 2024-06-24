import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, model, n_classes, training_layer=None):
        """
        Parameter
        ------------
        model : pre-trained model
        n_classes : output classes
        training_layer : list, choose which layer want to train. If want to train all model, set None
        """
        super().__init__()
        
        in_features = model.fc.in_features

        self.model = model
        self.model.fc = nn.Linear(in_features, n_classes)
        
        if training_layer:
            for name, param in self.model.named_parameters():
                param.requires_grad = False # free all parameters

                for stage in training_layer:
                    if stage in name:
                        param.requires_grad = True # unfreeze parameters if True
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    from simsiam import SimSiam
    from backbone import CustomizedResnet50

    simsiam = SimSiam(CustomizedResnet50())
    model = Classifier(model=simsiam.encoder[0].resnet, n_classes=1, training_layer=["layer4", "fc"])
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
        # print(name)
    
    
   