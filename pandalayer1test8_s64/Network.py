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


from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

# class MultiheadAttentionClassifier(nn.Module):
#     def __init__(self,num_classes,out_features,ninp,nhead,dropout,attention_dropout=0.1):
#         super(MultiheadAttentionClassifier, self).__init__()
#         #self.in_linear=nn.Linear(out_features,ninp)
#         self.attention=nn.MultiheadAttention(ninp, nhead, dropout=attention_dropout)
#         self.classifier=nn.Linear(ninp*2,num_classes)
#         self.dropout=nn.Dropout(dropout)
#         self.mish=Mish()
#
#     def forward(self,x):
#         #x=self.in_linear(x)
#         x=x.permute(1,0,2)
#         x,_=self.attention(x,x,x)
#         x=self.mish(x)
#         x=x.permute(1,0,2)
#         max_x,_=torch.max(x,dim=1)
#         x=torch.cat([torch.mean(x,dim=1),max_x],dim=-1)
#         #a=torch.max(x)
#         #print(a.shape)
#         #exit()
#         x=self.dropout(x)
#         x=self.classifier(x)
#         return x

class MultiheadAttentionClassifier(nn.Module):
    def __init__(self,num_classes,out_features,ninp,nhead,dropout,nlayers=1,attention_dropout=0.1):
        super(MultiheadAttentionClassifier, self).__init__()
        #self.in_linear=nn.Linear(out_features,ninp)
        #self.attention=nn.MultiheadAttention(ninp, nhead, dropout=attention_dropout)
        #print(ninp,nhead)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, ninp*2, attention_dropout)
        self.attention = nn.TransformerEncoder(encoder_layers, nlayers)
        self.classifier=nn.Linear(ninp*2,num_classes)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        #x=self.in_linear(x)
        x=self.dropout(x)
        x=x.permute(1,0,2)
        x=self.attention(x)
        x=x.permute(1,0,2)
        max_x,_=torch.max(x,dim=1)
        x=torch.cat([torch.mean(x,dim=1),max_x],dim=-1)
        #a=torch.max(x)
        #print(a.shape)
        #exit()
        x=self.dropout(x)
        x=self.classifier(x)
        return x



class Network(nn.Module):
    def __init__(self,num_classes,efficientNet_name='efficientnet-b0',dropout_p=0.5,nhead=8,ninp=256, nhid=512,attention_dropout=0.1):
        super(Network, self).__init__()
        #self.backbone=EfficientNet.from_pretrained(efficientNet_name,num_classes=6)
        self.backbone=models.resnet34(pretrained=True,)
        #self.backbone=models.resnet34(pretrained=True,)
        #self.backbone=torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        self.backbone.avgpool=GeM()
        #self.backbone.
        self.num_classes=num_classes
        self.out_features = self.backbone.fc.in_features
        self.ninp=ninp
        self.backbone.fc=nn.Identity()
        #self.backbone.fc=nn.Linear(self.out_features,ninp)
        regression_head=MultiheadAttentionClassifier(1,self.out_features,ninp,nhead,dropout_p)
        classifier_head=MultiheadAttentionClassifier(num_classes[0],self.out_features,ninp,nhead,dropout_p)
        regression_head2=MultiheadAttentionClassifier(1,self.out_features,ninp,nhead,dropout_p)
        regression_head3=MultiheadAttentionClassifier(1,self.out_features,ninp,nhead,dropout_p)

        self.heads=nn.ModuleList([regression_head,classifier_head,regression_head2,regression_head3,nn.Linear(self.out_features,ninp)])
        # for nclass in num_classes:
        #     self.heads.append(MultiheadAttentionClassifier(nclass,self.out_features,ninp,nhead,dropout_p))
        # self.heads=nn.ModuleList(self.heads)
        #self.mish=Mish()

    def forward(self,x):
        shape = x[0].shape
        bs=x.shape[0]
        n = len(x)
        x = x.reshape((-1,shape[1],shape[2],shape[3]))
        features=self.backbone(x)
        features=self.heads[4](features)
        features=features.reshape(bs,shape[0],self.ninp)
        outputs=[self.heads[0](features).squeeze(-1),F.log_softmax(self.heads[1](features),dim=1),
                 self.heads[2](features).squeeze(-1),self.heads[3](features).squeeze(-1)]
        # outputs=[]
        # for head in self.heads:
        #     outputs.append(F.log_softmax(head(features),dim=1))
        # print(outputs[0].shape)
        # print(outputs[1].shape)
        # print(outputs[2].shape)
        return outputs
