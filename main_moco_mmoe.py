from utils import *
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.cuda.amp import GradScaler

import moco.builder_mmoe as builder
import moco.loader
import moco.optimizer
import vit_moe as vits
from mytimer import mytimer
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names= vits.__all__+ torchvision_model_names

parser = argparse.ArgumentParser(description='MoCoV3')

parser.add_argument('--dist-url', default='tcp://localhost:10004', type=str)
parser.add_argument('--arch', metavar='ARCH', default='vit_col_112',choices=model_names)
parser.add_argument('--lr', default=3e-4, type=float,metavar='LR', dest='lr')
parser.add_argument('--name', default='', type=str,metavar='N')
parser.add_argument('--lamda', default='', type=float,metavar='N')

parser.add_argument('--data', default='/root/data/CICIOT_mini_split_8_0.5_0.3_0.2/train',metavar='DIR')
parser.add_argument('--workers', default=10, type=int, metavar='N')
parser.add_argument('--epochs', default=150, type=int, metavar='N')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument( '--weight-decay', default=0.1, type=float,metavar='W',dest='weight_decay')
parser.add_argument('--print-freq', default=100, type=int,metavar='N')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--multiprocessing-distributed', default=True, type=bool)
# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', default=False,type=bool)
parser.add_argument('--moco-t', default=0.2, type=float)

# vit specific configs:
parser.add_argument('--stop-grad-conv1', default=True, type=bool,help='stop grad conv1')
# other upgrades
parser.add_argument('--optimizer', default='adamw', type=str,choices=['lars', 'adamw'],)
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N')
parser.add_argument("--save_path", default='./model',help="Save the checkpoint file.")
parser.add_argument("--img_size", default=64, type=int,help="Save the checkpoint file.")
@mytimer
def main():
    time_prefix = datetime.now().strftime("%m-%d-%H")  # 例如 "04-03-14" 表示 4月3日14时
    args = parser.parse_args()
    save_dir=args.save_path+"/"+time_prefix+os.path.basename(os.path.normpath(args.data))+'_'+args.arch+args.name

    if not os.path.exists(args.save_path):#创建model文件夹
        os.makedirs(args.save_path, exist_ok=True)
    args.save_path = save_dir
    os.makedirs(save_dir, exist_ok=True)
    with open (save_dir+"/args.txt", "w") as f:
        f.write(str(args))
    with open (save_dir+"/lr_loss.csv", "w") as f:
        f.write("epoch,lr,loss\n")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    # Data loading code
    traindir =args.data
    config = load_config()
    data_key = args.data  # 使用完整路径作为键

    if data_key in config:
        print(f"Using precomputed stats for {data_key}")
        stats = config[data_key]
        train_images_mean = torch.Tensor(stats['mean'])
        train_images_std = torch.Tensor(stats['std'])
    else:
        print("Calculating mean and std of training images...")
        train_images_mean, train_images_std = calculate_mean_std(traindir, 128,4, 20000)
        print(f"Mean: {train_images_mean}, Std: {train_images_std}")
    # 保存新计算结果到配置
    config[data_key] = {
        'mean': train_images_mean.tolist(),
        'std': train_images_std.tolist()
    }
    save_config(config)
    print(f"Saved stats for {data_key} to config")
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1,img_size=args.img_size,classifier='None'),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
    else:
        model = builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm
    with open (args.save_path+"/args.txt", "a") as f:
        f.write(str(model))


    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler('cuda')
    summary_writer = SummaryWriter() if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    traindir =args.data
    config = load_config()
    data_key = args.data  # 使用完整路径作为键
    stats = config[data_key]
    train_images_mean = torch.Tensor(stats['mean'])
    train_images_std = torch.Tensor(stats['std'])
    
    normalize = transforms.Normalize(mean=train_images_mean,
                                     std=train_images_std)

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.Grayscale(num_output_channels=1),  # 强制转换为1通道灰度图
        transforms.ToTensor(),
        Transpose(),  # 使用自定义的转置操作
        normalize,
        RandomMask(p_masks=0.4, mask_size=(12, 4), p=0.2, image_size=args.img_size),
    ]

    augmentation2 = [
        #transforms.RandomApply([moco.loader.GaussianBlur([.1, 1.])], p=0.3),
        transforms.Grayscale(num_output_channels=1),  # 强制转换为1通道灰度图
        #transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
        transforms.ToTensor(),
        #Transpose(),  # 使用自定义的转置操作
        normalize,
        RandomMask(p_masks=0.4, mask_size=(12, 4), p=0.2, image_size=args.img_size),
    ]

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation1), 
                                      transforms.Compose(augmentation2)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,persistent_workers=True if args.workers > 0 else False,prefetch_factor=4 if args.workers > 0 else None,drop_last=True)
    total_samples = len(train_dataset)
    args.print_freq = max(1, total_samples // (args.batch_size*args.world_size* 5))  # 确保每个epoch打印大约5次
    print(f"Automatically set print frequency to {args.print_freq} (approx 5 prints per epoch)")
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)
        if(epoch==args.epochs):
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0): # only the first GPU saves checkpoint
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, 
                filename=args.save_path+"/"+os.path.basename(os.path.normpath(args.data))+'_'+args.arch+"_{:04d}.pth.tar".format(epoch)
            )

    if args.rank == 0:
        summary_writer.close()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            loss = model(images[0], images[1], moco_m,args.lamda)

        losses.update(loss.item(), images[0].size(0))
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    if args.rank == 0:
        with open(args.save_path + "/lr_loss.csv", "a") as f:
            f.write("{},{},{}\n".format(epoch, lr, loss.item()))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()
