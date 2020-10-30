import torch
from torch import nn

from efficientnet_pytorch import EfficientNet

'''
reference:
https://github.com/lukemelas/EfficientNet-PyTorch
'''

class predictor_EfficientNet(nn.Module):
    def __init__(self, img_size = (224,224), n_classes=42, version='b0', pretrain_backbone=True, freeze_backbone=True):
        super().__init__()
        
        self.n_classes = n_classes
        self.img_size = img_size
        self.version=version
        self.freeze_backbone = freeze_backbone
        
        # init process
        self.initBackbone(pretrain_backbone=pretrain_backbone)
        
        
    def initBackbone(self, pretrain_backbone=True):
        # check version
        if self.version == 'b0':
            # load backbone
            self._loadBackbone(version_name='efficientnet-b0', pretrain_backbone=pretrain_backbone)
            
            # init predictor
            self._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self._dropout = nn.Dropout(0.2)
            self._fc = nn.Linear(1280, self.n_classes)

        elif self.version == 'b3':
            # load backbone
            self._loadBackbone(version_name='efficientnet-b3', pretrain_backbone=pretrain_backbone)
            
            # init predictor
            self._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self._dropout = nn.Dropout(0.2)
            self._fc = nn.Linear(1280, self.n_classes)
        
        elif self.version == 'b7':
            # load backbone
            self._loadBackbone(version_name='efficientnet-b7', pretrain_backbone=pretrain_backbone)
            
            # init predictor
            self._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self._dropout = nn.Dropout(0.5)
            self._fc = nn.Linear(2560, self.n_classes)
        
        else :
            raise ValueError("[ERROR] unexpected version: %s, default \'b0\'" % self.version)
        
        
        self._freezeBackbone(freeze = self.freeze_backbone)
    
    # overwrite train()
    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
            
        self.backbone.train(not self.freeze_backbone) # train:False while freeze backbone is True --> eval()
        
        return self
        
    def forward(self, x):
        bs = x.shape[0] # batch size
        
        # feature extractor
        x = self.backbone.extract_features(x)
        
        # predictor
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        
        return x
    
    
    def _loadBackbone(self, version_name='efficientnet-b0', pretrain_backbone=True):
        if pretrain_backbone:
            self.backbone = EfficientNet.from_pretrained(version_name)
        else:
            self.backbone = EfficientNet.from_name(version_name)
        
    def _freezeBackbone(self, freeze=True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
            
#-------------------------- Main func ---------------------------#

def main():
    model = predictor_EfficientNet(pretrain_backbone=True, freeze_backbone=True)
    
    from torch.autograd import Variable
    x = Variable(torch.randn(1, 3, 224, 224))
    x = model(x)
    
    print('output shape :', x.shape)
    
if __name__ == '__main__':
    main()
