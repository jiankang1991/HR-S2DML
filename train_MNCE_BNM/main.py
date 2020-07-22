""" 
Normalized softmax with margin + BNM
"""
import os
import random
import math
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import shutil


import argparse
from tensorboardX import SummaryWriter


import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import sys
sys.path.append('../')

from utils.model import ResNet18, ResNet18_MLP
from utils.dataGen import DataGeneratorSM
from utils.metrics import MetricTracker, KNNClassification, NormSoftmaxLoss_Margin


parser = argparse.ArgumentParser(description='PyTorch SNCA Training for RS')
parser.add_argument('--data', metavar='DATA_DIR',  default='../data',
                        help='path to dataset (default: ../data)')
parser.add_argument('--dataset', metavar='DATASET',  default='ucmerced',
                        help='learning on the dataset (ucmerced)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--num_workers', default=8, type=int, metavar='N',
                        help='num_workers for data loading in pytorch, (default:8)')
parser.add_argument('--epochs', default=130, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--dim', default=128, type=int,
                    metavar='D', help='embedding dimension (default:128)')
parser.add_argument('--train_per', default=0.2, type=float,
                    metavar='Per', help='percentage of training data (default:0.2)')
parser.add_argument('--imgEXT', metavar='IMGEXT',  default='tif',
                        help='img extension of the dataset (default: tif)')
parser.add_argument('--temperature', default=0.05, type=float,
                    metavar='T', help='temperature (default:0.05)')
parser.add_argument('--MLP', dest='MLP', action='store_true',
                    help='use the nonlinear MLP')
parser.add_argument('--margin', default=0.5, type=float,
                    metavar='Margin', help='margin of the loss')

args = parser.parse_args()

sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
print('saving file name is ', sv_name)

checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
logs_dir = os.path.join('./', sv_name, 'logs')

if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.isdir(logs_dir):
    os.makedirs(logs_dir)

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))

def save_checkpoint(state, is_best, name):

    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth.tar')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name + '_model_best.pth.tar'))



def main():
    global args, sv_name, logs_dir, checkpoint_dir

    write_arguments_to_file(args, os.path.join('./', sv_name, sv_name+'_arguments.txt'))

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_data_transform = transforms.Compose([
                                        transforms.Resize((256,256)),
                                        transforms.RandomGrayscale(p=0.2),
                                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])

    test_data_transform = transforms.Compose([
                                            transforms.Resize((256,256)),
                                            transforms.ToTensor(),
                                            normalize])

    train_dataGen = DataGeneratorSM(data=args.data, 
                                    dataset=args.dataset,
                                    train_per=args.train_per, 
                                    imgExt=args.imgEXT,
                                    imgTransform=train_data_transform,
                                    phase='train')

    train_test_dataGen = DataGeneratorSM(data=args.data, 
                                    dataset=args.dataset,
                                    train_per=args.train_per, 
                                    imgExt=args.imgEXT,
                                    imgTransform=train_data_transform,
                                    phase='test')
    
    train_dataGen_ = DataGeneratorSM(data=args.data, 
                                    dataset=args.dataset,
                                    train_per=args.train_per,  
                                    imgExt=args.imgEXT,
                                    imgTransform=test_data_transform,
                                    phase='train')

    test_dataGen = DataGeneratorSM(data=args.data, 
                                    dataset=args.dataset,
                                    train_per=args.train_per, 
                                    imgExt=args.imgEXT,
                                    imgTransform=test_data_transform,
                                    phase='test')


    train_data_loader = DataLoader(train_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    train_test_data_loader = DataLoader(train_test_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_data_loader = DataLoader(test_dataGen, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    trainloader_wo_shuf = DataLoader(train_dataGen_, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    if args.MLP:
        model = ResNet18_MLP(dim = args.dim).cuda()
    else:
        model = ResNet18(dim = args.dim).cuda()

    loss_fn = NormSoftmaxLoss_Margin(args.dim, len(train_dataGen.sceneList), margin=args.margin, temperature=args.temperature).cuda()

    optimizer = torch.optim.SGD(list(model.parameters()) + list(loss_fn.parameters()), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4, nesterov=True)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_acc = 0
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['model_state_dict'])
            loss_fn.load_state_dict(checkpoint['loss_state_dict'])
            # lemniscate = checkpoint['lemniscate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))

    for epoch in range(start_epoch, args.epochs):

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        train(train_data_loader, train_test_data_loader, model, optimizer, loss_fn, epoch, train_writer)
        acc = val(test_data_loader, trainloader_wo_shuf, model, epoch, val_writer)

        is_best_acc = acc > best_acc
        best_acc = max(best_acc, acc)

        save_checkpoint({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'model_state_dict': model.state_dict(),
            'loss_state_dict': loss_fn.state_dict(), 
            # 'lemniscate': lemniscate,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best_acc, sv_name)

        scheduler.step()


def train(trainloader, train_test_loader, model, optimizer, MNCELoss, epoch, train_writer):

    global iter_train_test_loader

    total_losses = MetricTracker()
    MNCE_losses = MetricTracker()
    BNM_losses = MetricTracker()

    model.train()

    for idx, train_data in enumerate(tqdm(trainloader, desc="training")):
        iter_num = epoch * len(trainloader) + idx

        if iter_num % len(train_test_loader) == 0:
            iter_train_test_loader = iter(train_test_loader)
            
        train_imgs = train_data['img'].to(torch.device("cuda"))
        train_labels = train_data['label'].to(torch.device("cuda"))

        test_data = iter_train_test_loader.next()

        test_imgs = test_data['img'].to(torch.device("cuda"))
        test_labels = test_data['label'].to(torch.device("cuda"))

        train_e = model(train_imgs)

        _, mnceloss = MNCELoss(train_e, train_labels)

        test_e = model(test_imgs)

        testlogits, _ = MNCELoss(test_e, test_labels)
        softmax_tgt = nn.Softmax(dim=1)(testlogits)

        _, s_tgt, _ = torch.svd(softmax_tgt)
        transfer_loss = -torch.mean(s_tgt)

        total_loss = mnceloss + transfer_loss

        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()

        total_losses.update(total_loss.item(), train_imgs.size(0))
        MNCE_losses.update(mnceloss.item(), train_imgs.size(0))
        BNM_losses.update(transfer_loss.item(), train_imgs.size(0))

    info = {
        "Loss": total_losses.avg,
        "MNCE": MNCE_losses.avg,
        "BNM": BNM_losses.avg
    }

    for tag, value in info.items():
        train_writer.add_scalar(tag, value, epoch)

    print(f'Train TotalLoss: {total_losses.avg:.6f} NCE loss: {MNCE_losses.avg:.6f} BNM loss: {BNM_losses.avg:.6f}')



def val(valloader, trainloader_wo_shuf, model, epoch, val_writer):

    model.eval()
    
    train_features = []
    train_y_true = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(trainloader_wo_shuf, desc="extracting training data embeddings")):

            imgs = data['img'].to(torch.device("cuda"))
            label_batch = data['label'].to(torch.device("cpu"))

            e = model(imgs)

            train_features += list(e.cpu().numpy().astype(np.float32))
            train_y_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))

    train_features = np.asarray(train_features)
    train_y_true = np.asarray(train_y_true)

    knn_classifier = KNNClassification(train_features, train_y_true)

    y_val_true = []
    val_features = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="validation")):

            imgs = data['img'].to(torch.device("cuda"))
            label_batch = data['label'].to(torch.device("cpu"))

            e = model(imgs)

            val_features += list(e.cpu().numpy().astype(np.float32))
            y_val_true += list(np.squeeze(label_batch.numpy()).astype(np.float32))

    y_val_true = np.asarray(y_val_true)
    val_features = np.asarray(val_features)

    acc = knn_classifier(val_features, y_val_true)

    val_writer.add_scalar('KNN-Acc', acc, epoch)

    print('Validation KNN-Acc: {:.6f} '.format(
            acc,
            # hammingBallRadiusPrec.val,
            ))
    
    return acc

if __name__ == "__main__":
    main()


