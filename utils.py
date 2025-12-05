from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from mytimer import mytimer

import torch
import random
from collections import Counter

import json
import os

# 配置文件路径
CONFIG_PATH = '/home/ubuntu/work/Moco/dataset_stats.json'

def load_config():
    """加载配置文件，如果不存在或无效则返回空字典"""
    config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                # 检查是否为空文件
                if not config:
                    return {}
        except (json.JSONDecodeError, ValueError):
            print(f"警告: 配置文件 {CONFIG_PATH} 格式无效，将被重置")
            # 创建空文件或覆盖为有效JSON
            with open(CONFIG_PATH, 'w') as f:
                json.dump({}, f)
    return config

def save_config(config):
    """保存配置到文件"""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

# 添加计算类别权重的函数，放在main_worker函数之前
def calculate_class_weights(dataset, method='inverse'):
    """计算类别权重以处理类别不平衡问题
    
    Args:
        dataset: 数据集对象
        method: 权重计算方法，可选'inverse'(反比例),'sqrt_inverse'(平方根反比例),'effective_samples'(有效样本数)
    
    Returns:
        torch.Tensor: 每个类别的权重
    """
    # 获取所有标签
    targets = [sample[1] for sample in dataset.samples]
    class_counts = Counter(targets)
    num_classes = len(dataset.classes)
    
    # 确保所有类别都有计数
    counts = [class_counts.get(i, 0) for i in range(num_classes)]
    print(f"类别分布: {counts}")
    
    if method == 'inverse':
        # 使用类别频率的倒数作为权重
        weights = [1.0 / (count if count > 0 else 1.0) for count in counts]
    elif method == 'sqrt_inverse':
        # 使用类别频率平方根的倒数作为权重，减轻极端不平衡的影响
        weights = [1.0 / (np.sqrt(count) if count > 0 else 1.0) for count in counts]
    elif method == 'effective_samples':
        # 有效样本数方法 (Cui et al., 2019)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    else:
        raise ValueError(f"不支持的权重计算方法: {method}")
    
    # 归一化权重，使其和为num_classes
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes
    
    print(f"计算的类别权重: {weights}")
    return torch.FloatTensor(weights)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Transpose:
    def __call__(self, img):
        # 这里假设 img 是一个 Tensor, 需要转置
        # 对于 Tensor 的转置, x.transpose(0, 1) 会交换 height 和 width
        return img.transpose(1, 2)  # 将 H 和 W 交换（适用于 CHW 格式）


class RandomMask:
    def __init__(self, p_masks=0.1, mask_size=(8, 4), p=0.5, image_size=56):
        """
        :param p_masks: 需要生成遮挡区域的比例
        :param mask_size: 遮挡区域的尺寸 (height, width)
        :param p: 进行遮挡的概率
        :param image_size: 图像尺寸（假设是正方形）
        """
        self.image_size = int(image_size)
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.p_masks = float(p_masks)
        self.p = p

        # 计算最大 mask 数量
        self.num_masks = max(1, int((self.image_size ** 2) / (self.mask_size[0] * self.mask_size[1]) * self.p_masks))

    def __call__(self, img):
        """
        :param img: 输入图像，Tensor 形状 (C, H, W)
        :return: 遮挡后的图像
        """
        if random.random() > self.p:
            return img  # 以 (1-p) 概率不做遮挡

        _, H, W = img.shape

        # 生成随机 mask 坐标
        mask_h = torch.randint(1, self.mask_size[0] + 1, (self.num_masks,))
        mask_w = torch.randint(1, self.mask_size[1] + 1, (self.num_masks,))
        top = torch.randint(0, H - mask_h.max(), (self.num_masks,))
        left = torch.randint(0, W - mask_w.max(), (self.num_masks,))

        # 直接应用 mask
        for i in range(self.num_masks):
            img[:, top[i]:top[i] + mask_h[i], left[i]:left[i] + mask_w[i]] = 0

        return img


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def sanity_check(state_dict, pretrained_weights, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu",weights_only=False)
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword or '%s' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'module.base_encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

def plot_confusion_matrix(cm, class_names, save_path):
    """
    可视化并保存混淆矩阵。

    Parameters:
    - cm: 混淆矩阵 (numpy.ndarray)
    - class_names: 类别名称列表
    - save_path: 保存路径
    - dataset_name: 数据集名称
    - arch: 模型架构名称
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    # 显示标签
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()

    # 添加文本标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    # 保存混淆矩阵图片
    file_name = f"confusion_matrix.png"
    save_file_path = os.path.join(save_path, file_name)
    plt.savefig(save_file_path)
    plt.close()

@mytimer
def calculate_mean_std(data_dir, batch_size=128, num_workers=32, num_samples=20000, transform=None):
    # 如果没有提供transform，则设置默认的transform
    if transform is None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 强制转换为1通道灰度图
            transforms.ToTensor(),
        ])

    # 创建数据集和数据加载器
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 初始化计算mean和std的变量
    mean = 0.0
    std = 0.0
    total_images = 0

    # 遍历数据加载器，计算mean和std
    for i, (images, labels) in enumerate(data_loader):
        if total_images >= num_samples:
            break

        batch_samples = images.size(0)
        images = images.float()

        # 计算当前batch的mean和std
        batch_mean = images.mean(dim=(0, 2, 3))
        batch_std = images.std(dim=(0, 2, 3))

        # 更新总的mean和std
        mean += batch_mean * batch_samples
        std += batch_std * batch_samples
        total_images += batch_samples

    # 计算最终的mean和std
    mean /= total_images
    std /= total_images

    return mean, std

