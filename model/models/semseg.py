import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from .util import _BNReluConv, upsample


class SemsegModel(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True):
        super(SemsegModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.logits = _BNReluConv(self.backbone.num_features, self.num_classes, batch_norm=use_bn)

    def forward(self, pyramid): #, target_size, image_size):
        #data = {'pyramid': [pyramid],'target_size': (1024, 2048), 'target_size_feats': (1048 // 4, 2048 // 4), }
        #data = self.prepare_data(data, None)
        #pyramid = data['pyramid']
        pyramid = [p.clone().detach().requires_grad_(False).to('cuda') for p in [pyramid]]
        self.target_size = (256, 512)
        self.image_size = (1024, 2048)

        feats, additional0, additional1 = zip(*[self.backbone(p) for p in pyramid])
        #feature_pyramid = [upsample(f, self.target_size) for f in feats]
        feature_pyramid = [F.interpolate(f, self.target_size, mode='nearest') for f in feats]
        
        features = feature_pyramid[0] if len(feature_pyramid) == 1 else None
        logits = self.logits.forward(features)
        
        logits_i = F.interpolate(logits, self.image_size, mode='nearest')
        
        # x = torch.view(int(torch.argmax(logits_i, 1)))
        logits_ = torch.argmax(logits_i, 1) # calculates index of winning class  
        return logits_, additional0
    
        #return upsample(logits, self.image_size), additional0

    #def prepare_data(self, batch, image_size, device=torch.device('cuda')):
    #    if image_size is None:
    #        image_size = batch['target_size']
    #    pyramid = [p.clone().detach().requires_grad_(False).to(device) for p in batch['pyramid']]
    #    return {
    #        'pyramid': pyramid,
    #        'image_size': image_size,
    #        'target_size': batch['target_size_feats']
    #    }

    #def do_forward(self, batch, image_size=None):
    #    data = self.prepare_data(batch, image_size)
    #    return self.forward(**data)

    def random_init_params(self):
        return chain(*([self.logits.parameters(), self.backbone.random_init_params()]))

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()


class SemsegPyramidModel(nn.Module):
    def __init__(self, backbone, num_classes, use_bn=True, aux_logits=False):
        super(SemsegPyramidModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.num_features = self.backbone.features if isinstance(self.backbone.features,
                                                                 int) else self.backbone.num_features
        self.logits = _BNReluConv(self.num_features, self.num_classes, batch_norm=use_bn)
        self.has_aux_logits = aux_logits
        if aux_logits:
            self.num_features_aux = self.backbone.num_features_aux
            self.add_module('aux_logits', _BNReluConv(self.num_features_aux, self.num_classes, batch_norm=use_bn))

    def forward(self, pyramid, image_size):
        features, additional = self.backbone(pyramid)
        if self.has_aux_logits:
            additional['aux_logits'] = self.aux_logits.forward(additional['upsamples'][0])
        logits = self.logits.forward(features)
        return upsample(logits, image_size), additional

    def prepare_data(self, batch, image_size, device=torch.device('cuda')):
        if image_size is None:
            image_size = batch['target_size']
        pyr = [p.clone().detach().requires_grad_(False).to(device) for p in batch['pyramid']]
        return {'image_size': image_size, 'pyramid': pyr}

    def do_forward(self, batch, image_size=None):
        data = self.prepare_data(batch, image_size)
        return self.forward(**data)

    def random_init_params(self):
        params = [self.logits.parameters(), self.backbone.random_init_params()]
        if self.has_aux_logits:
            params += [self.aux_logits.parameters()]
        return chain(*params)

    def fine_tune_params(self):
        return self.backbone.fine_tune_params()
