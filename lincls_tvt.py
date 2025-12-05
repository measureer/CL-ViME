from utils import *

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import moco.loader

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score
import csv
#import vits
import torch.multiprocessing as mp
#import vits_ours as vits
import vit_moe2 as vits
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

mp.set_start_method('spawn', force=True)

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names= vits.__all__+ torchvision_model_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', default='CICIOT_mini_split_8_lincls',metavar='DIR')
parser.add_argument('--pretrained', default='model/04-16-20CICIOT_mini_split_8_vit_col_112/train_vit_col_112_0150.pth.tar', type=str)
parser.add_argument('--dist-url', default='tcp://localhost:29000', type=str) 
parser.add_argument('--batch-size', default=1024, type=int)
parser.add_argument('--val-batch-size', default=512, type=int)
parser.add_argument('--arch', metavar='ARCH', default='vit_col_112',choices=model_names)

parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)

parser.add_argument('--start-epoch', default=1, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=0., type=float,dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--evaluate',default=False, type=bool)
parser.add_argument('--world-size', default=1, type=int)
parser.add_argument('--rank', default=0, type=int)
parser.add_argument('--dist-backend', default='nccl', type=str)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--multiprocessing-distributed', default=True, type=bool)
parser.add_argument('--name', default='', type=str,metavar='N')

# additional configs:
parser.add_argument("--save_path", default='./model',help="Save the checkpoint file.")
parser.add_argument("--classifier", default='MLP', help="Classifier name.")
parser.add_argument('--optimizer', default='adamw', type=str,choices=['lars', 'adamw','sgd'])
parser.add_argument("--img_size", default=64, type=int,help="Save the checkpoint file.")
parser.add_argument('--weighted-loss', default=True, help='Use weighted loss for class imbalance')
parser.add_argument('--weight-method', default='inverse', type=str, choices=['inverse', 'sqrt_inverse', 'effective_samples'],
                    help='Method to calculate class weights: inverse, sqrt_inverse, or effective_samples')
best_acc1 = 0

def main():
    args = parser.parse_args()
    #创建目录
    lincls_dir = os.path.dirname(args.pretrained) + "/" + os.path.basename(os.path.normpath(args.data)) + '/'  + args.classifier+'_'+args.name
    if not os.path.exists(lincls_dir):
        os.makedirs(lincls_dir)
    if not os.path.exists(lincls_dir+'/best'):
        os.makedirs(lincls_dir+'/best')
    with open(lincls_dir+'/train_'+os.path.basename(os.path.normpath(args.data))+'.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
          # 写入表头（如果文件为空）
        if file.tell() == 0:
            writer.writerow(["Epoch", "Loss","Acc@1_avg"])
    with open(lincls_dir+'/val_'+os.path.basename(os.path.normpath(args.data))+'.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
          # 写入表头（如果文件为空）
        if file.tell() == 0:
            writer.writerow(["Epoch", "acc1","precision","recall","f1_score"])

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

    ngpus_per_node = torch.cuda.device_count()#每个设备上的GPU数量
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size#world_size总的GPU数量
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
   
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_model
    args.gpu = gpu
    lincls_dir = os.path.dirname(args.pretrained) + "/" + os.path.basename(os.path.normpath(args.data)) + '/'  + args.classifier+'_'+args.name
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        args.batch_size = int(args.batch_size / args.world_size)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')
    config = load_config()
    data_key = args.data  # 使用完整路径作为键
    stats = config[data_key]
    train_images_mean = torch.Tensor(stats['mean'])
    train_images_std = torch.Tensor(stats['std'])

    normalize = transforms.Normalize(mean=train_images_mean,
                                     std=train_images_std)
    aug1=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 强制转换为1通道灰度图
            transforms.ToTensor(),
            Transpose(),  # 使用自定义的转置操作
            normalize,
        ])
    aug2=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 强制转换为1通道灰度图
            transforms.ToTensor(),
            normalize,
        ])
    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(aug1,aug2))
    val_dataset = datasets.ImageFolder(
        valdir, moco.loader.TwoCropsTransform(aug1,aug2))
    test_dataset = datasets.ImageFolder(
        testdir, moco.loader.TwoCropsTransform(aug1,aug2))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    try:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler,persistent_workers=True if args.workers > 0 else False,prefetch_factor=4 if args.workers > 0 else None,drop_last=True)
    except Exception as e:
        print(f"Caught {e} when initializing DataLoader with persistent_workers and prefetch_factor. Trying without these parameters...")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,persistent_workers=True if args.workers > 0 else False,prefetch_factor=6 if args.workers > 0 else None,drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,persistent_workers=True if args.workers > 0 else False,prefetch_factor=6 if args.workers > 0 else None,drop_last=True)
    num_classes = len(train_dataset.classes)
    total_samples = len(train_dataset)
    args.print_freq = max(1, total_samples // (args.batch_size*args.world_size*5))  # 确保每个epoch打印大约5次
    print(f"Total samples: {total_samples}, Batch size: {args.batch_size},world_size: {args.world_size}")
    print(f"Automatically set print frequency to {args.print_freq} (approx 5 prints per epoch)")
    # create model
    print("=> creating model '{}'".format(args.arch))
    print
    model = vits.__dict__[args.arch](num_classes=num_classes,classifier=args.classifier,img_size=args.img_size)
    #print(model)
    linear_keyword = 'head'

    # freeze all layers but the head
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        nn.init.normal_(param, mean=0.0, std=0.01)    # init the head layer
        param.requires_grad = True
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            #print(state_dict.keys())
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):#head层的名称保持不动，为了不加载预训练的head层
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            
            #print(msg.missing_keys)
            assert all(k.startswith('head') for k in msg.missing_keys)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParall el will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.weighted_loss:
        print(f"使用带权重的损失函数，权重计算方法: {args.weight_method}")
        class_weights = calculate_class_weights(train_dataset, method=args.weight_method)
        if args.gpu is not None:
            class_weights = class_weights.cuda(args.gpu)
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    '''
    print("Parameters to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    '''
    #assert len(parameters) == 2  # weight, bias
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), init_lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    cudnn.benchmark = True
    print(model)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        loss,top1=train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1,precision, recall, f1, cm = validate(val_loader, model, criterion, args)


        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0): 
            #保存混淆矩阵
            cm_dir = os.path.join(lincls_dir, 'confusion_matrices')
            if not os.path.exists(cm_dir):
                os.makedirs(cm_dir, exist_ok=True)
            cm_path = os.path.join(cm_dir, f'epoch_{epoch}_cm.npy')
            np.save(cm_path, cm)
            with open(lincls_dir+'/train_'+os.path.basename(os.path.normpath(args.data))+'.csv', mode='a', newline='') as file1:
                writer = csv.writer(file1)
                writer.writerow([epoch,loss,top1])
            with open(lincls_dir+'/val_'+os.path.basename(os.path.normpath(args.data))+'.csv', mode='a', newline='') as file2:
                writer = csv.writer(file2)
                writer.writerow([
                    epoch,
                    round(acc1,4),
                    round(precision * 100, 4),  # 转换为百分比并保留两位小数
                    round(recall * 100, 4),  # 转换为百分比并保留两位小数
                    round(f1 * 100, 4)  # 转换为百分比并保留两位小数])
                ])
            # 保存最好的epoch
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_model=model
                plot_confusion_matrix(cm, val_loader.dataset.classes, lincls_dir+'/best')
                best_csv= lincls_dir+'/best/best_'+os.path.basename(os.path.normpath(args.data))+'.csv'
                with open(best_csv, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Epoch","Best_acc1", "Best_precision", "Best_recall", "Best_f1"])
                    writer.writerow([
                        epoch,
                        round(acc1,4),  # 转换 tensor 为 float
                        round(precision * 100, 4),  # 转换为百分比并保留两位小数
                        round(recall * 100, 4),  # 转换为百分比并保留两位小数
                        round(f1 * 100, 4)  # 转换为百分比并保留两位小数
                    ])
                #np.save(os.path.dirname(args.pretrained) + "/"+ os.path.basename(os.path.normpath(args.data))+'/'+args.name +'_cm.npy', cm)
                #保存模型
            '''
                save_checkpoint(
                    args,
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    filename=lincls_dir+"/" + os.path.basename(args.pretrained)
                )
            
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained, linear_keyword)
            '''
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0): 
        # evaluate on test set
        test_acc1,test_precision, test_recall, test_f1,test_cm = validate(test_loader, best_model, criterion, args)
        with open(lincls_dir+'/test.csv', mode='a', newline='') as file2:
            writer = csv.writer(file2)
            writer.writerow([
                epoch,
                round(test_acc1,4),
                round(test_precision * 100, 4),  # 转换为百分比并保留两位小数
                round(test_recall * 100, 4),  # 转换为百分比并保留两位小数
                round(test_f1 * 100, 4)  # 转换为百分比并保留两位小数])
            ])
        np.save(os.path.dirname(args.pretrained) + "/"+ os.path.basename(os.path.normpath(args.data))+'/'+args.name +'_test.npy', test_cm)
        print('Test accuracy: {:.4f}%'.format(test_acc1))

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images[0],images[1])
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1,acc5= accuracy(output, target, topk=(1,2))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.val, top1.avg.item()


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')
    # 用于存储所有预测和真实标签
    all_preds = []
    all_targets = []
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)

            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images[0],images[1])
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1,acc5 = accuracy(output, target, topk=(1,2))
            losses.update(loss.item(), images[0].size(0))
            top1.update(acc1[0], images[0].size(0))
            # 保存预测值和目标值
            _, preds = output.topk(1, 1, True, True)
            all_preds.append(preds.cpu())  # 保留Tensor
            all_targets.append(target.cpu())  # 保留Tensor
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        # 合并所有批次的预测和标签
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        # 转换为numpy数组
        all_preds = all_preds.numpy()
        all_targets = all_targets.numpy()
        # 计算Precision, Recall, F1
        acc=accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)

        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)

        # 输出结果
        print(f" * Acc@1 {top1.avg:.3f} Acc {acc:.4f}")
        print(f" * Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
        print(" * Confusion Matrix:")
        print(cm)

    return top1.avg.item(), precision, recall, f1, cm


def save_checkpoint(args,state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(args.pretrained) + "/" + os.path.basename(os.path.normpath(args.data)) + '/'  + args.classifier+"/best/model_best.pth.tar")

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

if __name__ == '__main__':
    main()
