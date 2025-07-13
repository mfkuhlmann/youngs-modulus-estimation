import torchvision.models as models
import torch
import torch.nn as nn

class VGG16FC7Extractor(nn.Module):
    def __init__(self):
        super(VGG16FC7Extractor, self).__init__()
    
        vgg16 = models.vgg16(pretrained=True)

        self.features = nn.Sequential(*list(vgg16.features.children()))
  
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc7 = nn.Sequential(*list(vgg16.classifier.children())[:-3])

    def forward(self, x):

        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        x = self.features(x)

        x = self.adaptive_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc7(x)
        return x