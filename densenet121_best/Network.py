from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

#mish activation
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))

class MultiheadAttentionClassifier(nn.Module):
    def __init__(self,num_classes,out_features,ninp,nhead,dropout,attention_dropout=0.1):
        super(MultiheadAttentionClassifier, self).__init__()
        self.in_linear=nn.Linear(out_features,ninp)
        self.attention=nn.MultiheadAttention(ninp, nhead, dropout=attention_dropout)
        self.classifier=nn.Linear(ninp,num_classes)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        x=self.in_linear(x)
        x=x.permute(1,0,2)
        x,_=self.attention(x,x,x)
        x=x.permute(1,0,2)
        x=torch.mean(x,dim=1)
        x=self.dropout(x)
        x=self.classifier(x)
        return x


class Network(nn.Module):
    def __init__(self,num_classes,efficientNet_name='efficientnet-b0',dropout_p=0.5,nhead=8,ninp=512, nhid=512,attention_dropout=0.1):
        super(Network, self).__init__()
        #self.backbone=EfficientNet.from_pretrained(efficientNet_name,num_classes=6)
        self.backbone=models.resnext50_32x4d(pretrained=True)
        self.num_classes=num_classes
        self.out_features = self.backbone.fc.in_features
        self.ninp=ninp
        self.backbone.fc=nn.Identity()
        self.heads=[]
        for nclass in num_classes:
            self.heads.append(MultiheadAttentionClassifier(nclass,self.out_features,ninp,nhead,dropout_p))
        self.heads=nn.ModuleList(self.heads)
        #self.mish=Mish()

    def forward(self,x):
        shape = x[0].shape
        bs=x.shape[0]
        n = len(x)
        x = x.reshape((-1,shape[1],shape[2],shape[3]))
        features=self.backbone(x)
        features=features.reshape(bs,shape[0],self.out_features)
        outputs=[]
        for head in self.heads:
            outputs.append(F.log_softmax(head(features),dim=1))
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # print(outputs[2].shape)
        return outputs
