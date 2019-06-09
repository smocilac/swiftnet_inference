import torch
import torch.onnx 

from torch.autograd import Variable
from models.resnet.resnet_single_scale import *
from models.semseg import SemsegModel
import numpy as np
import time

PATH_PARAMS = '/<path to params>'
INPUT_RESOLUTION = [1,3, 1024, 2048]

if __name__=='__main__':
    use_bn = True
    resnet = resnet18(pretrained=False, efficient=False, use_bn=use_bn)
    model = SemsegModel(resnet, 19, use_bn=use_bn)
    model.load_state_dict(torch.load(PATH_PARAMS), strict=True)
    model.to('cuda') 
    input_ = torch.ones(INPUT_RESOLUTION).to('cuda')
    n = 1000    
    with torch.no_grad():
        #input = model.prepare_data(batch, conf.target_size, device=device) # sends to GPU
        logits = model.forward(input_) # performs inference
        t0 = 1000 * time.perf_counter()
        torch.cuda.synchronize()
        for _ in range(0, n):
            logits = model.forward(input_)
            #_, pred = logits.max(1) # calculates index of winning class
            #out = pred.data.byte().cpu() # transfers only 8bit indices of winning classes to CPU
        torch.cuda.synchronize()
        t1 = 1000 * time.perf_counter()
    t = (t1 - t0)
    fps = (1000 * n) / t
    print (fps)

