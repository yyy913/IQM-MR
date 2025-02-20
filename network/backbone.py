import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50, densenet121, inception_v3

class BackboneV4(nn.Module):
    def __init__(self, cfg):
        super(BackboneV4, self).__init__()
        if cfg.backbone == 'resnet50':
            self.encoder = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.encoder.avgpool = nn.Identity()
            self.encoder.fc = nn.Identity()

            self.reduce_dim = nn.Sequential(
                nn.Conv2d(2048, cfg.reduced_dim, kernel_size=1, padding=0),
                nn.AdaptiveAvgPool2d(1),
            )

            self.reduce_dim[0].weight.data.normal_(0, 0.01)
            self.reduce_dim[0].bias.data.zero_()

        else:
            raise ValueError(f'[!] undefined backbone architecture has been given: {cfg.backbone}.')

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.backbone_name = cfg.backbone

    def _forward(self, im):

        f1 = self.encoder.maxpool(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(im))))
        f2 = self.encoder.layer1(f1)
        f3 = self.encoder.layer2(f2)
        f4 = self.encoder.layer3(f3)
        f5 = self.encoder.layer4(f4)

        base_embs = self.reduce_dim(f5)

        return base_embs

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        model_dict = {
            'resnet50': (resnet50, "RadImageNet_pytorch/ResNet50.pt"),
            'densenet121': (densenet121, "RadImageNet_pytorch/DenseNet121.pt"),
            'inception_v3': (inception_v3, "RadImageNet_pytorch/InceptionV3.pt")
        }
        
        if cfg.backbone not in model_dict:
            raise ValueError(f'[!] undefined backbone architecture: {cfg.backbone}')
        
        model_fn, weight_path = model_dict[cfg.backbone]
        
        if cfg.backbone == 'inception_v3':
            self.encoder = model_fn(pretrained=False, aux_logits=False)
            self.backbone = nn.Sequential(
                self.encoder.Conv2d_1a_3x3,
                self.encoder.Conv2d_2a_3x3,
                self.encoder.Conv2d_2b_3x3,
                self.encoder.maxpool1,
                self.encoder.Conv2d_3b_1x1,
                self.encoder.Conv2d_4a_3x3,
                self.encoder.maxpool2,
                self.encoder.Mixed_5b,
                self.encoder.Mixed_5c,
                self.encoder.Mixed_5d,
                self.encoder.Mixed_6a,
                self.encoder.Mixed_6b,
                self.encoder.Mixed_6c,
                self.encoder.Mixed_6d,
                self.encoder.Mixed_6e,
                self.encoder.Mixed_7a,
                self.encoder.Mixed_7b,
                self.encoder.Mixed_7c,
            )
        elif cfg.backbone == 'densenet121':
            self.encoder = model_fn(pretrained=False)
            self.backbone = self.encoder.features
        else: 
            self.encoder = model_fn(pretrained=False)
            layers = list(self.encoder.children())
            self.backbone = nn.Sequential(*layers[:8])
        
        state_dict = torch.load(weight_path)
        new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        if cfg.backbone == 'densenet121':
            new_state_dict = {(k[2:] if k.startswith("0.") else k): v for k, v in new_state_dict.items()}
        self.backbone.load_state_dict(new_state_dict)
        
        if cfg.backbone in ['resnet50', 'inception_v3']:
            num_channels = 2048
        elif cfg.backbone == 'densenet121':
            num_channels = 1024
        
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(num_channels, cfg.reduced_dim, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d(1)
        )
        self.reduce_dim[0].weight.data.normal_(0, 0.01)
        self.reduce_dim[0].bias.data.zero_()
                        
    def _forward(self, x):
        features = self.backbone(x)
        reduced = self.reduce_dim(features)
        return reduced