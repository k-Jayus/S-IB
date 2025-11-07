import argparse
import os
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from models import *

from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate
import random
import numpy as np
import torch.nn.functional as F
def set_seed(seed=42):
    """固定随机种子以确保实验可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 使用方法
set_seed(43)  # 可以改成任何你想要的种子值
torch.cuda.set_device(3)
model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0.0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    import torchvision.models as models

    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features  # 获取输入特征数 (512)
    model.fc = nn.Linear(num_features, 120)  # 修改为100类
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    # cudnn.benchmark = True
    #stanford dogs
    from torchvision.datasets.utils import download_and_extract_archive
    from torch.utils.data import Dataset
    from scipy.io import loadmat
    from PIL import Image
    # Data loading
    class StanfordDogs(Dataset):
        def __init__(self, root='./data', train=True, transform=None, download=False):
            self.root = os.path.join(root, 'StanfordDogs')
            self.train = train
            self.transform = transform

            if download:
                self.download()

            # 读取train/test列表
            split_file = os.path.join(self.root, 'train_list.mat' if train else 'test_list.mat')
            mat_data = loadmat(split_file)
            file_list = mat_data['file_list']
            labels = mat_data['labels'].flatten() - 1  # MATLAB索引从1开始,转为0开始

            self.data = []
            for i, file_path in enumerate(file_list):
                img_path = os.path.join(self.root, 'Images', file_path[0][0])
                self.data.append((img_path, labels[i]))

        def download(self):
            if os.path.exists(self.root):
                print("Dataset already exists, skipping download.")
                return

            # 下载图片
            images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
            download_and_extract_archive(images_url, self.root, filename='images.tar')

            # 下载列表文件
            lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
            download_and_extract_archive(lists_url, self.root, filename='lists.tar')

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path, label = self.data[idx]
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label

    # Stanford Dogs
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = StanfordDogs(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=97,
        shuffle=True,
        num_workers=2
    )

    testset = StanfordDogs(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=97,
        shuffle=False,
        num_workers=2
    )

    print(f"Original Stanford Dogs test set size: {len(testset)}")
    print(f"Number of classes: 120")
    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)


        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, args.arch + '_dogs-1.pth')



# --------------------------------------------------
# 0. 互信息估计器
# --------------------------------------------------
def hsic_unbiased(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC with linear kernel, O(B²)."""
    B = z1.size(0)
    K = z1 @ z1.T
    L = z2 @ z2.T
    H = torch.eye(B, device=z1.device) - 1. / B
    Kc = H @ K @ H
    Lc = H @ L @ H
    return (Kc * Lc).sum() / (B - 1) ** 2



@torch.no_grad()
def _shuffle_rows(t: torch.Tensor) -> torch.Tensor:
    return t[torch.randperm(t.size(0))]


# --------------------------------------------------
# 1. 掩码生成
# --------------------------------------------------
def mask_adaptive(S: torch.Tensor, temperature: float = 10.,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    你的 max-norm + mean-center + sigmoid 版本。
    S: (B,1,H,W) or (B,C,H,W)
    """
    S_norm = S / (S.amax(dim=(1, 2, 3), keepdim=True) + eps)
    return torch.sigmoid(
        temperature * (S_norm - S_norm.mean(dim=(1, 2, 3), keepdim=True))
    )

def mask_root(S: torch.Tensor, p: float = 0.1, α: float = 30.,
              iters: int = 6, eps: float = 1e-6) -> torch.Tensor:
    """
    “可导 top-p” ：求 t 使 mean(sigmoid(α(S-t))) = p.
    完全无排序，数值稳定。
    """
    B = S.size(0)
    S_norm = S / (S.amax(dim=(1,2,3), keepdim=True) + eps)
    with torch.no_grad():
        t = torch.full((B,1,1,1), 0.5, device=S.device)
        for _ in range(iters):
            m = torch.sigmoid(α*(S_norm - t))
            t = t + 0.5 * (m.mean(dim=(1,2,3), keepdim=True) - p)
    return torch.sigmoid(α*(S_norm - t))
def batch_integral_density_mask_v2(image_tensor, window_size=31, device='cuda'):
    """
    自适应阈值版本，避免全为1的问题
    """
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.tensor(image_tensor, device=device, dtype=torch.float32)
    else:
        image_tensor = image_tensor.to(device)

    batch_size, channels, h, w = image_tensor.shape

    # 转换为灰度图
    gray = torch.mean(image_tensor, dim=1)
    gray_abs = torch.abs(gray)

    # 批量归一化
    gray_flat = gray_abs.view(batch_size, -1)
    gray_min = torch.quantile(gray_flat, 0.05, dim=1, keepdim=True).view(batch_size, 1, 1)
    gray_max = torch.quantile(gray_flat, 0.95, dim=1, keepdim=True).view(batch_size, 1, 1)
    gray_norm = (gray_abs - gray_min) / (gray_max - gray_min + 1e-8)

    # 自适应阈值：基于75%分位数
    gray_norm_flat = gray_norm.view(batch_size, -1)
    adaptive_threshold = torch.quantile(gray_norm_flat, 0.8, dim=1).view(batch_size, 1, 1)

    # 高强度区域
    high_intensity = (gray_norm > adaptive_threshold).float()

    # 后续代码不变...
    input_tensor = high_intensity.unsqueeze(1)
    density = F.avg_pool2d(input_tensor,
                           kernel_size=window_size,
                           stride=1,
                           padding=window_size // 2)
    density = density.squeeze(1)

    density_flat = density.view(batch_size, -1)
    thresholds = torch.quantile(density_flat, 0.8, dim=1).view(batch_size, 1, 1)
    mask = (density >= thresholds).float()

    return mask
def min_max_normalize(x, eps=1e-10):
    """
    最大最小归一化到[0,1]

    Args:
        x: 输入张量，形状为 [n, 3*224*224]
        eps: 防止除零

    Returns:
        归一化后的张量 [n, 3*224*224]
    """
    # 每个样本独立缩放到[0,1]
    x_min = x.amin(dim=(0,1,2,3), keepdim=True)
    x_max = x.amax(dim=(0,1,2,3), keepdim=True)

    return (x - x_min) / (x_max - x_min + eps)
import pytorch_msssim
def ssim_loss(img1, img2):
    """
    SSIM作为损失函数（越小越相似）
    """
    ssim_value = pytorch_msssim.ssim(img1, img2, data_range=1)
    return 1 - ssim_value  # 转换为loss


def differentiable_quantile(x, q, dim=-1, temperature=1.0):
    """
    可微分的分位数近似计算
    x: 输入张量
    q: 分位数 (0-1之间)
    dim: 计算分位数的维度
    temperature: 温度参数，越小越接近真实分位数
    """
    # 对输入进行排序
    sorted_x, _ = torch.sort(x, dim=dim)

    # 计算分位数对应的索引位置
    n = x.shape[dim]
    target_idx = q * (n - 1)

    # 将target_idx转换为张量
    target_idx = torch.tensor(target_idx, device=x.device, dtype=torch.float32)

    # 使用softmax加权来近似选择对应位置的值
    indices = torch.arange(n, device=x.device, dtype=torch.float32)

    if dim != -1:
        # 调整indices的形状以匹配目标维度
        indices_shape = [1] * len(x.shape)
        indices_shape[dim] = n
        indices = indices.view(indices_shape)

        # 调整target_idx的形状
        target_shape = [1] * len(x.shape)
        target_shape[dim] = 1
        target_idx = target_idx.view(target_shape)

    # 计算距离并应用softmax
    distances = -torch.abs(indices - target_idx) / temperature
    weights = F.softmax(distances, dim=dim)

    # 加权求和得到分位数近似值
    quantile_approx = torch.sum(sorted_x * weights, dim=dim, keepdim=True)

    return quantile_approx


def batch_integral_density_mask_v2_differentiable(image_tensor, window_size=31, device='cuda',
                                                  temperature=0.1, slope=10.0):
    """
    可微分版本的自适应阈值函数
    temperature: 控制分位数近似的精度，越小越精确
    slope: 控制sigmoid阈值的陡峭程度，越大越接近硬阈值
    """
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.tensor(image_tensor, device=device, dtype=torch.float32)
    else:
        image_tensor = image_tensor.to(device)

    batch_size, channels, h, w = image_tensor.shape

    # 转换为灰度图
    gray = torch.mean(image_tensor, dim=1)
    gray_abs = torch.abs(gray)

    # 批量归一化 - 使用可微分的分位数计算
    gray_flat = gray_abs.view(batch_size, -1)
    gray_min = differentiable_quantile(gray_flat, 0.05, dim=1, temperature=temperature).view(batch_size, 1, 1)
    gray_max = differentiable_quantile(gray_flat, 0.95, dim=1, temperature=temperature).view(batch_size, 1, 1)
    gray_norm = (gray_abs - gray_min) / (gray_max - gray_min + 1e-8)

    # 自适应阈值：基于80%分位数 - 使用可微分版本
    gray_norm_flat = gray_norm.view(batch_size, -1)
    adaptive_threshold = differentiable_quantile(gray_norm_flat, 0.8, dim=1, temperature=temperature).view(batch_size,
                                                                                                           1, 1)

    # 高强度区域 - 使用sigmoid软阈值替代硬阈值
    high_intensity = torch.sigmoid((gray_norm - adaptive_threshold) * slope)

    # 后续代码逻辑保持不变
    input_tensor = high_intensity.unsqueeze(1)
    density = F.avg_pool2d(input_tensor,
                           kernel_size=window_size,
                           stride=1,
                           padding=window_size // 2)
    density = density.squeeze(1)

    # 使用可微分的分位数计算阈值
    density_flat = density.view(batch_size, -1)
    thresholds = differentiable_quantile(density_flat, 0.8, dim=1, temperature=temperature).view(batch_size, 1, 1)

    # 使用sigmoid软阈值替代硬阈值
    mask = torch.sigmoid((density - thresholds) * slope)

    return mask
def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_mu= AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    mi_type = "hsic"
    hsic_rand_feat: int | None = None
    # switch to train mode
    model.train()
    mse_loss = torch.nn.MSELoss()
    end = time.time()
    for i, (x, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        y = y.cuda(non_blocking=True)

        x = x.cuda(non_blocking=True).requires_grad_(True)
        B = x.size(0)
        # compute output
        optimizer.zero_grad()

        logits = model(x)
        ce = F.cross_entropy(logits, y)

        # ---------- (1)  gradient × output ----------

        p = logits.softmax(dim=1)
        p2_sum = (p ** 2).sum() * 0.5  # ½‖p‖² , grad = Jᵀp
        R = torch.autograd.grad(p2_sum, x, create_graph=True)[0]
        #p = logits.softmax(dim=1)
        '''p2_sum = (logits).sum()   # ½‖p‖² , grad = Jᵀp
        R = torch.autograd.grad(p2_sum, x, create_graph=True)[0]'''


        '''p2_sum = (logits).sum()   # ½‖p‖² , grad = Jᵀp
        R = torch.autograd.grad(p2_sum, x, create_graph=True)[0]'''
        #M = batch_integral_density_mask_v2(R).unsqueeze(1)
        M = batch_integral_density_mask_v2_differentiable(R).unsqueeze(1)


        # ---------- (4) 前景对齐 ----------

        fore_R = F.normalize((R * (M)).flatten(1), dim=1)
        fore_x = F.normalize((x * (M)).flatten(1), dim=1)
        if mi_type == "hsic":
            if hsic_rand_feat is not None:
                proj = torch.randn(fore_R.size(1), hsic_rand_feat, device='cuda')
                fore_R = F.normalize(fore_R @ proj, dim=1)
                fore_x = F.normalize(fore_x @ proj, dim=1)
            align_loss = hsic_unbiased(fore_R, fore_x)
        else:
            raise ValueError(mi_type)
        '''fore_R = min_max_normalize((R * (M)))
        fore_x = min_max_normalize((x * (M)))
        align_loss = ssim_loss(fore_R,fore_x)'''


        # ---------- (5) 背景互信息 ----------

        '''back_R = F.normalize((R * (1 - M)).flatten(1), dim=1)
        back_x = F.normalize((x * (1 - M)).flatten(1), dim=1)

        if mi_type == "hsic":
            if hsic_rand_feat is not None:
                proj = torch.randn(back_R.size(1), hsic_rand_feat, device='cuda')
                back_R = F.normalize(back_R @ proj, dim=1)
                back_x = F.normalize(back_x @ proj, dim=1)
            mi_bg = hsic_unbiased(back_R, back_x)
        else:
            raise ValueError(mi_type)'''
        #mi_bg = torch.mean(torch.abs(R) * (1 - M))
        mi_bg = torch.mean(torch.var(((R * (1-M))),dim=(1,2,3)))
        #mi_bg = torch.mean(torch.var(((R * (1 - M))), dim=(1, 2, 3)))

        # ---------- (6) total ----------

        #loss = ce + mi_bg * 0.01+align_loss*0.001
        loss = ce + torch.exp(mi_bg) * 1000 + torch.exp(-align_loss) * 0.01
        #0.01 0.001 0.1 0.0001
        x = x.detach()

        prec1, prec5 = accuracy(logits.data, y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        loss_mu.update(mi_bg.item()/align_loss.item(), x.size(0))
        top1.update(prec1[0], x.size(0))
        top5.update(prec5[0], x.size(0))

        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'loss_mu {loss_mu.val:.4f} ({loss_mu.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses,loss_mu=loss_mu, top1=top1, top5=top5))
        torch.cuda.empty_cache()


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
