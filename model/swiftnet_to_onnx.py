import torch
import torch.onnx 

from torch.autograd import Variable
from models.resnet.resnet_single_scale import *
from models.semseg import SemsegModel
import numpy as np

if __name__=='__main__':
    use_bn = True
    resnet = resnet18(pretrained=False, efficient=False, use_bn=use_bn)
    model = SemsegModel(resnet, 19, use_bn=use_bn)
    model.load_state_dict(torch.load('/home/smocilac/dipl_seminar/swiftnet/weights/swiftnet_ss_cs.pt'), strict=True)
    model.to('cuda') 
    input_ = torch.ones([1,3,256,512]).to('cuda')
    model.forward(input_)
    #torch.onnx.export(model, input_, "swiftnet.onnx", verbose=True, do_constant_folding=True)

