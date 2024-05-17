# import torch相关文件
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
# import 非torch相关文件
import numpy as np
import random
import ast
import os
import argparse
import time
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack, GradientSignAttack
# from advertorch.attacks import CarliniWagnerLinfAttack
from advertorch.context import ctx_noparamgrad_and_eval

# import 自己定义的文件
from utils import Logger
from models import *
from autoattack import AutoAttack

# 固定随机数保证可复现
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(1)


# =======================训练前准备========================
# ===========定义并创建路径，设置GPU，数据集和模型===========
parser = argparse.ArgumentParser(description='IAD-I')
# path parameter
parser.add_argument('--S_path', type=str, default='./models/student/', help='student_model_path')
parser.add_argument('--T_path', type=str, default='./models/teacher/', help='teacher_model_path')
parser.add_argument('--logs_path', type=str, default='./logs/', help='logs_path')

# input parameter
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--epochs', type=int, default=200, help='epochs')

# model parameter
parser.add_argument('--model', type=str, default='ResNet18', help='model')
parser.add_argument('--T_model', type=str, default='ResNet18', help='model')
# optimizer parameter
parser.add_argument('--lr', type=int, default=0.1, help='initial learning rate')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150], help='Decrease learning rate at these epochs')
parser.add_argument('--lr_factor', default=0.1, type=float, help='factor by which to decrease lr')
parser.add_argument('--weight_decay', type=int, default=0.0002, help='weight_decay')

# loss parameter
parser.add_argument('--lam', type=float, default=1.0, help='lambda')

# attack parameter
parser.add_argument('--type', type=str, default='PGD', help='attack name')
parser.add_argument('--epsilon', type=float, default=8/255, help='attack norm')
parser.add_argument('--iter_train', type=int, default=10, help='attack iterations during training')
parser.add_argument('--iter_test', type=int, default=20, help='attack iterations during testing')
parser.add_argument('--step', type=float, default=2/255, help='single step')

# training parameter
args = parser.parse_args()

# 选定训练过程中的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
device = torch.device("cuda:0")



# make sure that every path exists
# if not os.path.exists(args.data):
#     os.makedirs(args.data)

if not os.path.exists(args.S_path):
    os.makedirs(args.S_path)

if not os.path.exists(args.T_path):
    os.makedirs(args.T_path)

if not os.path.exists(args.logs_path):
    os.makedirs(args.logs_path)

# dataset
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='~/data/cifar-10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='~/data/cifar-10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_classes = 10
elif args.dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='~/data/cifar-100', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR100(root='~/data/cifar-100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_sizes, shuffle=False, num_workers=2)
    num_classes = 100

# T_model and S_model
print('==> Building model..'+args.model)
# student
if args.model == 'MobileNetV2':
	net = ResNet18(num_classes=num_classes)
elif args.model == 'WideResNet':
	net = ResNet18(num_classes=num_classes)
elif args.model == 'ResNet18':
	net = ResNet18(num_classes=num_classes)
net = net.to(device)
# net.load_state_dict(torch.load(os.path.join(args.S_path, 'checkpoint_104_IAD.pth'))['state_dict'])

# teacher
if args.T_model == 'MobileNetV2':
    teacher_net = ResNet18(num_classes=num_classes)
elif args.T_model == 'WideResNet':
    teacher_net = ResNet18(num_classes=num_classes)
elif args.T_model == 'ResNet18':
    teacher_net = ResNet18(num_classes=num_classes)
teacher_net = teacher_net.to(device)
for param in teacher_net.parameters():
    param.requires_grad = False

print('==> Loading teacher..')
teacher_net.load_state_dict(torch.load(os.path.join(args.T_path, 'bestpoint.pth'))['state_dict'])
teacher_net.eval()

# loss
KL_loss = nn.KLDivLoss(reduction='none')
XENT_loss = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_schedule, gamma=args.lr_factor)

# tensorboard or logger
writer = SummaryWriter(os.path.join(args.logs_path,args.dataset,args.model+'-adv/'))

logger_ctrl = Logger(os.path.join(args.logs_path, 'ctrl.txt'), title='ctrl')
logger_ctrl.set_names(['Epoch', 'batch_idx', 'ctrl_LI'])

adversary = LinfPGDAttack(net, loss_fn=XENT_loss, eps=args.epsilon, nb_iter=args.iter_train, eps_iter=args.step, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary_test = LinfPGDAttack(net, loss_fn=XENT_loss, eps=args.epsilon, nb_iter=args.iter_test, eps_iter=args.step, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
# adversary_test = AutoAttack(net, norm='Linf', eps=args.epsilon, version='standard')
# adversary_test = GradientSignAttack(net, loss_fn=XENT_loss, eps=args.epsilon, targeted=False)
# adversary_test = CarliniWagnerLinfAttack(net, max_iterations=1, num_classes=num_classes)
def train(epoch, optimizer, net, teacher_net, adversary):
    
    # 开始计时并设置进度条
    torch.cuda.synchronize()
    start = time.time()
    iterator = tqdm(trainloader, ncols=0, leave=False)

    # 将net调为train模式
    net.train()

    # 定义训练中需要用到的常量
    train_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(iterator):

        # 加载batch迭代中需要的数据
        inputs, targets = inputs.to(device), targets.to(device)
        Alpha = torch.zeros(len(inputs)).cuda()

        # optimizer 梯度置零  
        optimizer.zero_grad()

        # 获取输出
        with ctx_noparamgrad_and_eval(net):
            pert_inputs = adversary.perturb(inputs, targets)
        
        outputs = net(pert_inputs)
        clean_outputs = net(inputs)

        teacher_outputs = teacher_net(inputs)
        guide = teacher_net(pert_inputs)

        # 计算loss
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
        # targets_onehot_clean = F.one_hot(targets, num_classes=num_classes).float()
        # targets_onehot_clean[targets_onehot > 0.5] = 0.91
        # targets_onehot_clean[targets_onehot < 0.5] = 0.01
        if epoch <= 40:
            targets_onehot[targets_onehot > 0.5] = 0.91
            targets_onehot[targets_onehot < 0.5] = 0.01
        elif 40 <= epoch <= 200:
            targets_onehot[targets_onehot > 0.5] = 0.82
            targets_onehot[targets_onehot < 0.5] = 0.02

	    
        ctrl_LI = 0.1 * (KL_loss(F.log_softmax(teacher_outputs, dim=1), targets_onehot) + KL_loss(F.log_softmax(guide, dim=1), targets_onehot)).sum(dim=1)
        loss = args.lam*(1/len(outputs))*torch.sum(KL_loss(F.log_softmax(outputs, dim=1),F.softmax(teacher_outputs, dim=1)).sum(dim=1)) + args.lam*(1/len(outputs))*torch.sum(KL_loss(F.log_softmax(outputs, dim=1),F.softmax(net(inputs), dim=1)).sum(dim=1).mul(ctrl_LI))+(1.0-args.lam)*XENT_loss(net(inputs), targets)
        
        # 计算梯度
        loss.backward()

        # 梯度回传
        optimizer.step()

        # 计算batch数据并记录在进度条中
        # logger_ctrl.append([epoch + 1, batch_idx + 1, ctrl_LI.sum()])

        train_loss += loss.item()
        iterator.set_description(str(loss.item()))
    
    # 计算epoch数据
    torch.cuda.synchronize()
    end = time.time()
    end_start = end-start
    logger_ctrl.append([epoch + 1, 0, train_loss/len(iterator)])
    print(end-start)
    print('Mean Training Loss:', train_loss/len(iterator))
    return end_start, train_loss

def main():

    # 定义main函数中需要的常量
    best_acc = 0

    for epoch in range(args.epochs):

        # 定义main函数中需要的常量
        print("teacher >>>> student ")

        # train
        T_end_start, _ = train(epoch, optimizer, net, teacher_net, adversary)

        epoch结束后更新学习率
        scheduler.step()

        # 保存模型文件
        if epoch > 99:
            torch.save({
                        'epoch': epoch + 1,
                        'state_dict': net.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, os.path.join(args.S_path, 'checkpoint_%d.pth'%epoch))     
            

if __name__ == '__main__':
    main()
