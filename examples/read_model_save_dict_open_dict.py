import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
import collections
import torch.optim as optim
from torchvision import transforms
from torch.nn.modules.container import Sequential


from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

# set when prune_train
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 7"

def get_dataloader():
    dataset_train = CocoDataset("/home/xiongyizhe/pytorch-retinanet/coco", set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset("/home/xiongyizhe/pytorch-retinanet/coco", set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    return dataset_train, dataset_val

target_model_name = "model_after_prune_round1.pth"
retinanet = torch.load(target_model_name)

print(retinanet)

torch.save(retinanet.module.state_dict(), "retina_after_prune.pt")

dataset_train, dataset_val = get_dataloader()
sampler = AspectRatioBasedSampler(dataset_train, batch_size=4, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

if dataset_val is not None:
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=4, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

from retinanet import model
retinanet_read = model.resnet50(num_classes=dataset_train.num_classes(),)
checkpoint = torch.load("retina_after_prune.pt")
retinanet_read.load_state_dict(checkpoint)

print(retinanet_read)

coco_eval.evaluate_coco(dataset_val, retinanet_read)

