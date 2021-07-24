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

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'prune_train', 'test', 'continue_train'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--epoch', type=int, default=10)

args = parser.parse_args()

def get_dataloader():
    dataset_train = CocoDataset("/home/xiongyizhe/pytorch-retinanet/coco", set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset("/home/xiongyizhe/pytorch-retinanet/coco", set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    return dataset_train, dataset_val

def eval(retinanet, dataset_val):
    coco_eval.evaluate_coco(dataset_val, retinanet)

def train_model(retinanet, dataloader_train, dataset_val):
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=3e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()
    
    for epoch_num in range(args.epoch):
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                with open("res_prune_trainlog_round{}.txt".format(int(args.round)), 'a') as f:
                    f.writelines('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} \n'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                
                del classification_loss
                del regression_loss

                if (args.mode=="prune_train") and iter_num % 5000 == 0:
                    torch.save(retinanet, 'last_iter_train.pt')
            except Exception as e:
                print(e)
                continue

        print('Evaluating dataset')
        coco_eval.evaluate_coco(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet, 'model_after_prune_round{}.pth'.format(int(args.round)))
        torch.save(retinanet, 'last_epoch_train.pt')

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')


def prune_model(retinanet):
    for m in retinanet.module.modules():
        m = m.cpu()
    DG = tp.DependencyGraph().build_dependency( retinanet.module, torch.randn(1, 3, 512, 512) )
    def prune_conv(conv, amount=0.2):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    # pass
    block_prune_probs = [0.1, 0.2, 0.2, 0.3]
    blk_id = 0
    for m in retinanet.modules():
        if isinstance(m, Sequential) and len(m) > 2:  # m is a BottleNeck
            for bottleneck in m:
                print(bottleneck.conv1)
                prune_conv( bottleneck.conv1, block_prune_probs[blk_id] )
                prune_conv( bottleneck.conv2, block_prune_probs[blk_id] )
                prune_conv( bottleneck.conv3, block_prune_probs[blk_id] )
            blk_id += 1

    n = retinanet.module.regressionModel
    prune_conv( n.conv1, block_prune_probs[0] )
    prune_conv( n.conv2, block_prune_probs[1] )
    prune_conv( n.conv3, block_prune_probs[2] )
    prune_conv( n.conv4, block_prune_probs[3] )
    n = retinanet.module.classificationModel
    prune_conv( n.conv1, block_prune_probs[0] )
    prune_conv( n.conv2, block_prune_probs[1] )
    prune_conv( n.conv3, block_prune_probs[2] )
    prune_conv( n.conv4, block_prune_probs[3] )

    return retinanet

def main():

    # do not use if prune
    dataset_train, dataset_val = get_dataloader()
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=4, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=4, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    
    from retinanet import model
    retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)

    # do not use if prune or prune_train
    # retinanet = retinanet.cuda()
    # retinanet.load_state_dict(torch.load("/home/xiongyizhe/pytorch-retinanet/coco_resnet_50_map_0_335_state_dict.pt"))
    # if torch.cuda.is_available():
    #     retinanet = retinanet.cuda()
    

    if args.mode=='train':
        if torch.cuda.is_available():
            retinanet = torch.nn.DataParallel(retinanet).cuda()
        else:
            retinanet = torch.nn.DataParallel(retinanet)
        print('Num training images: {}'.format(len(dataset_train)))
        train_model(retinanet, dataloader_train, dataset_val)
    elif args.mode=='prune':
        # pass
        previous_ckpt = 'model_after_prune_round{}.pth'.format(int(args.round)-1)
        print("Pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        retinanet = torch.load(previous_ckpt, map_location=torch.device('cpu'))
        # retinanet = retinanet.module
        # retinanet.load_state_dict( torch.load(previous_ckpt, map_location=torch.device('cpu')) )
        # print(retinanet)
        prune_model(retinanet)
        torch.save(retinanet, 'model_prune_round{}.pth'.format(int(args.round)))
        print(retinanet)
        exit(0)
        # params = sum([np.prod(p.size()) for p in model.parameters()])
        # print("Number of Parameters: %.1fM"%(params/1e6))
        if torch.cuda.is_available():
            retinanet = torch.nn.DataParallel(retinanet).cuda()
        else:
            retinanet = torch.nn.DataParallel(retinanet)
        train_model(retinanet, dataloader_train, dataset_val)
    elif args.mode=='prune_train':
        retinanet = retinanet.cuda()
        previous_ckpt = 'model_prune_round{}.pth'.format(int(args.round))
        # previous_ckpt = 'last_iter_train.pt'
        print("Train pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        retinanet = torch.load(previous_ckpt).module
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
        if torch.cuda.is_available():
            retinanet = torch.nn.DataParallel(retinanet).cuda()
        else:
            retinanet = torch.nn.DataParallel(retinanet)
        train_model(retinanet, dataloader_train, dataset_val)
        # pass
    elif args.mode=='test':
        ckpt = 'model_after_prune_round{}.pth'.format(args.round)
        print("Load model from %s"%( ckpt ))
        retinanet = torch.load(ckpt).module
        eval(retinanet, dataset_val)
    elif args.mode=='continue_train':
        previous_ckpt = 'model_after_prune_round{}.pth'.format(int(args.round))
        print("Train pruning round %d, load model from %s"%( args.round, previous_ckpt ))
        retinanet = torch.load(previous_ckpt).module
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()
        if torch.cuda.is_available():
            retinanet = torch.nn.DataParallel(retinanet).cuda()
        else:
            retinanet = torch.nn.DataParallel(retinanet)
        train_model(retinanet, dataloader_train, dataset_val)

if __name__=='__main__':
    main()
