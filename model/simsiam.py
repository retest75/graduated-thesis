# handcraft ResNet-50 model used to be as backbone
# it obey the ImageNet mode having 1000-classes output

from torch import nn
from torchvision.models import resnet50
import torch

class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # encoder寫法(1): 可以看出完整的網路架構
        #backbone.fc.out_features = 2048
        self.encoder = nn.Sequential(backbone, Projection())
        
        # encoder寫法(2): 打印網路架構不對，但不影響輸出結果
        #self.backbone = backbone
        #self.backbone.fc.out_features = 2048                               # backbone
        #self.projection = Projection(in_dim=self.backbone.fc.out_features) # 3-layer projection MLP
        #self.encoder = nn.Sequential(self.backbone, self.projection)       # encoder

        # 2-layer predictor MLP
        self.predictor = Predictor()

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()
        

class Bottleneck(nn.Module):
    """ Create a bottleneck block in ResNet-50 
    Parameters
    ------------
    in_channels  : int, input channel of previous bottleneck
    out_channels : int, output channel of first convolution in current block
    stride       : default = 1, however if this is the second conv in first block and layer1 to layer4, it must be 2
    downsample   : sequential, if size or dimension is different, it must downsample

    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        expansion = 4 # the output channel of last convolution must expand 4 times
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels*expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x # use to skip connection

        if self.downsample is not None:
            identity = self.downsample(x)

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        output += identity # residual connection
        output = self.relu(output)

        return output

class ResNet50(nn.Module):
    """ Create the ResNet-50 architecture
    Attribute
    ------------
    self.channels : input channels to first conv in each layer, default = 64 (layer 1)

    Parameter
    ------------
    num_classes : output classes
    """
    def __init__(self, num_classes=2048):
        super().__init__()
        self.in_channels = 64

        # first conv layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #layer 1 to 4
        self.layer1 = self._make_layer(Bottleneck, channel=64, num_block=3)
        self.layer2 = self._make_layer(Bottleneck, channel=128, num_block=4 ,stride=2)
        self.layer3 = self._make_layer(Bottleneck, channel=256, num_block=6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, channel=512, num_block=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=512*4, out_features=num_classes)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output

    def _make_layer(self, block, channel, num_block, stride=1):
        """ Create each layer composed of some Bottleneck
        Parameter
        ------------
        block : module, a  bottleneck
        channel : output channels for first conv layer in each layer (當前layer的第一個conv的輸出維度)
        num_block : int, number of block in each layer
        stride : int, indicate stride size to conv2 in first block of each layer

        Return
        ------------
        Sequential type
        """
        downsample = None

        # if skip dimension or size is different from bottleneck outputm then downsampling
        # it happens to first block in each layer
        if (stride != 1) or (self.in_channels != channel * 4):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=channel*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=channel * 4)
            )
        
        ##### the first conv layer in each bottleneck #####
        # 1. 因為第一個 bottleneck 結束，輸出後的維度是輸入的4倍
        # 2. 所以第二個 bottleneck 開始前，輸入的維度要是輸出的4倍
        # 3. 所以 self.in_channels 需要 *4
        ###################################################
        layer = []
        layer.append(block(in_channels=self.in_channels, out_channels=channel, stride=stride, downsample=downsample))
        self.in_channels = channel * 4
        ###################################################

        for _ in range(1, num_block):
            layer.append(block(in_channels=self.in_channels, out_channels=channel))

        return nn.Sequential(*layer)

class Projection(nn.Module):
    """ 
    3- layer projection MLP in encoder 
    Each fc layer do not use bias since it followed by BN
    """

    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class Predictor(nn.Module):
    """
    2-layer prodictor MLP in SimSiam 
    Each fc layer do not use bias since it followed bu BN
    """
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


if __name__ == "__main__":
    #model = ResNet50()
    #model = SimSiam(ResNet50())
    model = SimSiam(resnet50())
    print(model)
    

    #encoder = nn.Sequential(model)
    #print(encoder)