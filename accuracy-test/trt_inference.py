import os
import importlib.util
import torch
from pathlib import Path

from data.transform import *
from data.mux.transform import *
from torchvision.transforms import Compose

import numpy as np
from data.cityscapes import Cityscapes
import numpy as np
from tqdm import tqdm
from time import perf_counter
import lib.cylib as cylib
from torch.utils.data import DataLoader
from data.cityscapes import Cityscapes
import subprocess


def compute_errors(conf_mat, class_info, verbose=True):
    num_correct = conf_mat.trace()
    num_classes = conf_mat.shape[0]
    total_size = conf_mat.sum()
    avg_pixel_acc = num_correct / total_size * 100.0
    TPFP = conf_mat.sum(0)
    TPFN = conf_mat.sum(1)
    FN = TPFN - conf_mat.diagonal()
    FP = TPFP - conf_mat.diagonal()
    class_iou = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_precision = np.zeros(num_classes)
    per_class_iou = []
    if verbose:
        print('Errors:')
    for i in range(num_classes):
        TP = conf_mat[i, i]
        class_iou[i] = (TP / (TP + FP[i] + FN[i])) * 100.0
        if TPFN[i] > 0:
            class_recall[i] = (TP / TPFN[i]) * 100.0
        else:
            class_recall[i] = 0
        if TPFP[i] > 0:
            class_precision[i] = (TP / TPFP[i]) * 100.0
        else:
            class_precision[i] = 0

        class_name = class_info[i]
        per_class_iou += [(class_name, class_iou[i])]
        if verbose:
            print('\t%s IoU accuracy = %.2f %%' % (class_name, class_iou[i]))
    avg_class_iou = class_iou.mean()
    avg_class_recall = class_recall.mean()
    avg_class_precision = class_precision.mean()
    if verbose:
        print('IoU mean class accuracy -> TP / (TP+FN+FP) = %.2f %%' % avg_class_iou)
        print('mean class recall -> TP / (TP+FN) = %.2f %%' % avg_class_recall)
        print('mean class precision -> TP / (TP+FP) = %.2f %%' % avg_class_precision)
        print('pixel accuracy = %.2f %%' % avg_pixel_acc)
    return avg_pixel_acc, avg_class_iou, avg_class_recall, avg_class_precision, total_size, per_class_iou


def volume(shape):
    vol = 1
    for i in shape:
        vol *= i
    return vol


def run_acc_test( data_loader, class_info):
    conf_mat = np.zeros((Cityscapes.num_classes, Cityscapes.num_classes), dtype=np.uint64)
    simPath = "./../build/"
    imageName = "image.txt"
    
    for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch['original_labels'] = batch['original_labels'].numpy().astype(np.uint32)
        h_input = batch['pyramid'][0].flatten().numpy()
        h_input.tofile(imageName)
        args = (simPath + "tensorrt_test", "-m", "1", "-e", simPath + "engine.trt", "-r", os.path.dirname(os.path.realpath(__file__)) + "/" + imageName)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        h_output = np.fromfile("image.txt.out", dtype=np.float32)
        h_output = h_output.reshape((19,1024,2048))
        #pred = torch.argmax(h_output, dim=1).byte().numpy().astype(np.uint32)
        pred = np.argmax(h_output, axis=0).astype(np.uint32)
        #import pdb; pdb.set_trace()
        cylib.collect_confusion_matrix(pred.flatten(), batch['original_labels'].flatten(), conf_mat)
        

    print('')
    pixel_acc, iou_acc, recall, precision, _, per_class_iou = compute_errors(conf_mat, class_info, verbose=True)

    return iou_acc, per_class_iou

alphas = [1.]
target_size = ts = (2048, 1024)
target_size_feats = (ts[0] // 4, ts[1] // 4)
scale = 255
mean = Cityscapes.mean
std = Cityscapes.std
nw = 1

trans_train = trans_val = Compose(
    [Open(),
     RemapLabels(Cityscapes.map_to_id, Cityscapes.num_classes),
     Pyramid(alphas=alphas),
     SetTargetSize(target_size=target_size, target_size_feats=target_size_feats),
     Normalize(scale, mean, std),
     Tensor(),
     ]
)

if __name__ == "__main__":
    root = Path('datasets/Cityscapes')
    dataset_val = Cityscapes(root, transforms=trans_val, subset='val')
    loader_val = DataLoader(dataset_val, batch_size=1, collate_fn=custom_collate, num_workers=nw)
    eval_loaders = [(loader_val, 'val')]

    class_info = dataset_val.class_info
    

    for loader, name in eval_loaders:
        iou, per_class_iou = run_acc_test( loader, class_info )
        print(f'{name}: {iou:.2f}')
