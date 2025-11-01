"""
遥感数据集加载模块
支持数据集：AID, NaSC, NWPU, PatternNet, UCM
"""
import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from dataset_config import get_dataset_config


# ==================== 路径工具函数 ====================

def get_path_to_project_root():
    """
    获取项目根目录路径
    兼容Windows和Linux路径
    
    Returns:
        str: 项目根目录的绝对路径
    """
    cwd = os.path.abspath(os.getcwd())
    # 查找 JEIDNN 目录
    path = cwd
    while path != os.path.dirname(path):  # 直到到达根目录
        if os.path.basename(path) == "JEIDNN":
            return path
        path = os.path.dirname(path)
    
    # 如果没有找到JEIDNN，返回当前目录
    return cwd


def get_abs_path(paths_strings):
    """
    根据相对路径字符串列表构建绝对路径
    
    Args:
        paths_strings: 路径字符串列表，如 ['checkpoint', 'subfolder']
        
    Returns:
        str: 绝对路径
    """
    subpath = os.sep.join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return os.path.join(src_abs_path, subpath)


def split_dataloader_in_n(data_loader, n):
    """
    将数据加载器分割成n个子加载器
    
    Args:
        data_loader: 原始数据加载器
        n: 分割数量
        
    Returns:
        list: n个数据加载器的列表
    """
    try:
        indices = data_loader.sampler.indices
    except:
        indices = list(range(len(data_loader.dataset)))
    dataset = data_loader.dataset
    list_indices = np.array_split(np.array(indices), n)
    batch_size = data_loader.batch_size
    n_loaders = []
    for i in range(n):
        sampler = SubsetRandomSampler(list_indices[i])
        sub_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler
        )
        n_loaders.append(sub_loader)
    return n_loaders


# ==================== 数据集构建函数 ====================


def build_dataset(is_train, args, infer_no_resize=False):
    """
    构建遥感数据集（固定划分：train/val）
    支持：AID, NWPU, PatternNet, UCM
    """
    transform = build_transform(is_train, args, infer_no_resize)
    dataset_config = get_dataset_config(args.data_set)

    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {args.data_set}. Supported datasets: AID, NaSC, NWPU, PatternNet, UCM")
    
    if dataset_config['use_random_split']:
        raise ValueError(f"Dataset {args.data_set} uses random split. Please use build_dataset_new instead.")
    
    # 使用配置文件统一处理遥感数据集
    if dataset_config['train_folder'] is not None:
        folder = dataset_config['train_folder'] if is_train else dataset_config['val_folder']
        root = os.path.join(args.data_path, folder)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = dataset_config['num_classes']
    else:
        raise ValueError(f"Dataset {args.data_set} configuration is incomplete")

    return dataset, nb_classes

def build_dataset_new(args, seed=0, split_ratio=0.8, infer_no_resize=False):
    """
    使用随机划分方式构建数据集
    支持：NaSC
    """
    transform_train = build_transform(True, args, infer_no_resize)
    transform_test = build_transform(False, args, infer_no_resize)
    root = args.data_path
    dataset_config = get_dataset_config(args.data_set)

    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {args.data_set}. Supported datasets: AID, NaSC, NWPU, PatternNet, UCM")
    
    if not dataset_config['use_random_split']:
        raise ValueError(f"Dataset {args.data_set} uses fixed split. Please use build_dataset instead.")
    
    # 使用配置文件处理遥感数据集
    dataset = datasets.ImageFolder(root, transform=transform_train)
    nb_classes = dataset_config['num_classes']

    # 随机划分数据集
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(seed)
    )

    # 更新测试集的transform
    test_dataset.dataset.transform = transform_test

    return train_dataset, test_dataset, nb_classes


def build_transform(is_train, args, infer_no_resize=False):
    if hasattr(args, 'arch'):
        if 'cait' in args.arch and not is_train:
            print('# using cait eval transform')
            transformations = {}
            transformations= transforms.Compose(
                [transforms.Resize(args.input_size, interpolation=3),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
            return transformations
    
    if infer_no_resize:
        print('# using cait eval transform')
        transformations = {}
        transformations= transforms.Compose(
            [transforms.Resize(args.input_size, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        return transformations

    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
