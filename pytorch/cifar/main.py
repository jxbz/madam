'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import numpy as np

from utils import progress_bar

#cifar-100
from utils_cifar100 import get_training_dataloader, get_test_dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--beta_2', default=0.9, type=float)
parser.add_argument('--gamma', default=0, type=float)
parser.add_argument('--bsz', default=1024, type=int)
parser.add_argument('--lr_decay_steps', default=100, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--SGD', action='store_true')
parser.add_argument('--MUL', action='store_true')
parser.add_argument('--MUL_experimental', action='store_true')
parser.add_argument('--signSGD', action='store_true')
parser.add_argument('--signSGDW', action='store_true')
parser.add_argument('--fromage', action='store_true')
parser.add_argument('--cifar100', action='store_true')
parser.add_argument('--logdir', default='results/tensorboard/test', type=str)
parser.add_argument('--prior_dir', default='no_prior', type=str)
parser.add_argument('--w_clipping', default=0, type=float)
parser.add_argument('--resume_dir', default='no_resume', type=str)


#log knobs
parser.add_argument('--log_accumulated_sign', action='store_true')
parser.add_argument('--log_dist_w', action='store_true')

#train enviorment
parser.add_argument('--dgx', action='store_true')

#compression knobs
parser.add_argument('--init_w_quant_level', default=0, type=float)
parser.add_argument('--init_w_scale', default=0, type=float)
parser.add_argument('--model_dir', default='no_prior', type=str)
parser.add_argument('--dynamic_quant', action='store_true')
parser.add_argument('--modified_net', action='store_true')

#limiting states
parser.add_argument('--num_states', default=0, type=int)
parser.add_argument('--penalty_alpha', default=0, type=float)
parser.add_argument('--dynamic_range', default=0, type=float)

#multi_ladders
parser.add_argument('--multi_ladder', action='store_true')
parser.add_argument('--divider', default=1, type=int)
parser.add_argument('--signsgd', action='store_true')

#inc_ladder
parser.add_argument('--inc_ladder', action='store_true')

parser.add_argument('--scale', default=3, type=float)
parser.add_argument('--lr_madam', default=0.01, type=float)
parser.add_argument('--lr_adam', default=0.01, type=float)

#wandb
parser.add_argument('--name', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--group', type=str)
parser.add_argument('--ngc', action='store_true')

#new added
parser.add_argument('--lr_factor', default=1, type=int)
parser.add_argument('--num_levels', default=1, type=int)

#new optimizer
parser.add_argument('--madam', action='store_true')
parser.add_argument('--int_madam', action='store_true')
parser.add_argument('--adam', action='store_true')
parser.add_argument('--decay_gamma', default=0.1, type=float)


args = parser.parse_args()

import sys
sys.path.append('..')

from optim.madam import Madam
from optim.int_madam import IntegerMadam


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if not args.cifar100:
    #cifar-10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bsz, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bsz, shuffle=False, num_workers=2)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes=100
    print('==> Start training on CIFAR-10')
else:
    #cifar-100
    TRAIN_SETTING_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    TRAIN_SETTING_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    trainloader = get_training_dataloader(TRAIN_SETTING_MEAN, TRAIN_SETTING_STD, batch_size=args.bsz, shuffle=True, num_workers=2)
    testloader = get_test_dataloader(TRAIN_SETTING_MEAN, TRAIN_SETTING_STD, batch_size=args.bsz,shuffle=False, num_workers=2)
    num_classes=10
    print('==> Start training on CIFAR-100')


# Model
if not args.cifar100:
    from models import *
else:
    from models_c100 import *
print('==> Building model..')
# net = VGG('VGG19')

#set affine to False in BN for Resnet
if not args.modified_net:
    print('start normal')
    net = ResNet18()
else:
    print('start modified')
    net = ResNet18_modified()

# net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
if args.SGD:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0) #wd 5e-4
elif args.madam:
    optimizer = Madam(net.parameters(), lr=args.lr, p_scale=args.scale)
elif args.int_madam:
    optimizer = IntegerMadam(net.parameters(), base_lr=args.lr, lr_factor=args.lr_factor, levels=args.num_levels, p_scale=args.scale)
elif args.adam:
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.resume_dir is not 'no_resume':
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.resume_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume_dir + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['opt_state'])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=0.1)
def lr_schedule(optimizer, epoch, step_size=100, gamma=0.1, divider=10, multi_ladder=False):
    if epoch % args.lr_decay_steps == 0:
        print('start lr decay')
        for group in optimizer.param_groups:
            if args.int_madam:
                group['lr_factor'] = group['lr_factor'] / divider
                if group['lr_factor'] < 1:
                    group['lr_factor'] = 1
            else:
                group['lr'] = group['lr'] * gamma


test(start_epoch)
for epoch in range(start_epoch, start_epoch+args.epochs):
    epoch += 1
    train(epoch)
    test(epoch)
    lr_schedule(optimizer, epoch, step_size=args.lr_decay_steps, gamma=args.decay_gamma, divider=args.divider, multi_ladder=args.multi_ladder)
