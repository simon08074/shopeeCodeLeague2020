import torch
from torch import nn
import torchvision.models as models

'''
reference:

# torchvision
https://github.com/pytorch/vision 

# pretrain model as backbone
https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18

'''

class predictor_ResNet(nn.Module):
    def __init__(self, img_size = (224,224), n_classes=42, version='18', pretrain_backbone=True):
        super().__init__()
        
        self.n_classes = n_classes
        self.img_size = img_size
        self.backbone = None # type : nn.Module
        self.extracted_features_flatten_dim = None # type: int
        
        if isinstance(version, int):
            version = str(version)
        self.version=version
        
        # init process
        self.initBackbone(pretrain_backbone=pretrain_backbone)
        
        
    def initBackbone(self, pretrain_backbone=True):
        # check version
        if self.version == '18':
            self.extracted_features_flatten_dim = 512
            
            # load backbone
            self.backbone = models.resnet18(pretrained = pretrain_backbone)
            self.backbone.extract_features = torch.nn.Sequential(*list(self.backbone.children())[:-1]) # exclude fc layer

            # init predictor
            self._fc = nn.Linear(self.extracted_features_flatten_dim, self.n_classes)
        
        # check version
        elif self.version == '34':
            self.extracted_features_flatten_dim = 512
            
            # load backbone
            self.backbone = models.resnet34(pretrained = pretrain_backbone)
            self.backbone.extract_features = torch.nn.Sequential(*list(self.backbone.children())[:-1]) # exclude fc layer

            # init predictor
            self._fc = nn.Linear(self.extracted_features_flatten_dim, self.n_classes)
            
        # check version
        elif self.version == '50':
            self.extracted_features_flatten_dim = 2048
            
            # load backbone
            self.backbone = models.resnet50(pretrained = pretrain_backbone)
            self.backbone.extract_features = torch.nn.Sequential(*list(self.backbone.children())[:-1]) # exclude fc layer

            # init predictor
            self._fc = nn.Linear(self.extracted_features_flatten_dim, self.n_classes)
            
        # check version
        elif self.version == '101':
            self.extracted_features_flatten_dim = 2048
            
            # load backbone
            self.backbone = models.resnet101(pretrained = pretrain_backbone)
            self.backbone.extract_features = torch.nn.Sequential(*list(self.backbone.children())[:-1]) # exclude fc layer

            # init predictor
            self._fc = nn.Linear(self.extracted_features_flatten_dim, self.n_classes)
            
        # check version
        elif self.version == '152':
            self.extracted_features_flatten_dim = 2048
            
            # load backbone
            self.backbone = models.resnet152(pretrained = pretrain_backbone)
            self.backbone.extract_features = torch.nn.Sequential(*list(self.backbone.children())[:-1]) # exclude fc layer

            # init predictor
            self._fc = nn.Linear(self.extracted_features_flatten_dim, self.n_classes)
            
        else :
            raise ValueError("[ERROR] unexpected version: '\%s'\, default \'18\'" % self.version)
        
        self._freezeBackbone()
        
    # overwrite train()
    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
            
        # set backbone to eval mode   
        self.backbone.train(False) # eval()
        
        return self
        
        
    def forward(self, x):
        bs = x.shape[0] # batch size
        
        # feature extractor
        x = self.backbone.extract_features(x)
        x = x.view(-1, self.extracted_features_flatten_dim)
        
        # predictor
        x = self._fc(x)
        
        return x
        
    def _freezeBackbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
            
#-------------------------- Main func ---------------------------#

def main():
    model = predictor_ResNet(version='152', pretrain_backbone=False)
    
    from torch.autograd import Variable
    x = Variable(torch.randn(1, 3, 224, 224))
    x = model(x)
    
    print('output shape :', x.shape)
    
    
if __name__ == '__main__':
    main()
