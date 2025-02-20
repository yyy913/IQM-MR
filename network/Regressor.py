import torch.nn as nn
from network.backbone import BackboneV4

class Regressor(nn.Module):
    def __init__(self, cfg):
        super(Regressor, self).__init__()
        self.backbone = BackboneV4(cfg)
        self.regressor = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(cfg.reduced_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone._forward(x)  
        output = self.regressor(features)      
        return output