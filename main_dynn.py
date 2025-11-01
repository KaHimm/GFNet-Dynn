'''Train DYNN from checkpoint of trained backbone'''
import sys
sys.path.append("/workspace/JEIDNN/dynn")

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from timm.models import *
from dynn.op_counter import measure_model_and_assign_cost_per_exit
from datasets import get_path_to_project_root
from dynn.learning_helper import freeze_backbone as freeze_backbone_helper, LearningHelper
from dynn.log_helper import setup_mlflow
from dynn.classifier_training_helper import LossContributionMode
from dynn.gate.gate import GateType
from dynn.gate_training_helper import GateObjective
#/home/Dynn/workspace/data

from dynn.our_train_helper import set_from_validation, evaluate, train_single_epoch, eval_baseline, dynamic_warmup, test_layer, perform_test
from datasets import build_dataset, build_dataset_new
from dynn.gfnet_dynn import GFNet_xs_dynn
from functools import partial
import torch.nn as nn
from datetime import datetime
from dataset_config import get_dataset_config

now = datetime.now()                # current date and time


parser = argparse.ArgumentParser(
    description='JEI-DNN with GFNet Backbone for Remote Sensing Image Classification')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--arch', type=str,
                    choices=['GFNet-xs-dynn'], # baseline is to train only with warmup, no gating
                    default='GFNet-xs-dynn', help='model to train'
                    )
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr',default=2e-4,type=float,help='minimal learning rate')
parser.add_argument('--input-size', default=224, type=int, help='images input size')

# # Dataset parameters
parser.add_argument('--data-path', default=None, type=str,
                        help='dataset path (required for remote sensing datasets)')
parser.add_argument('--data-set', default='UCM', choices=['UCM', 'NaSC', 'PatternNet', 'AID', 'NWPU'],
                        type=str, help='Remote sensing dataset name (AID, NaSC, NWPU, PatternNet, UCM)')
parser.add_argument('--checkpoint-path', default=None, type=str,
                        help='Path to pretrained model checkpoint (required for training/evaluation)')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
parser.add_argument('--repeated-aug', action='store_true')
parser.set_defaults(repeated_aug=False)
parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--ce_ic_tradeoff',default=0.75,type=float,help='cost inference and cross entropy loss tradeoff')
parser.add_argument('--num_epoch', default=200, type=int, help='num of epochs')
parser.add_argument('--max_warmup_epoch', default=10, type=int, help='max num of warmup epochs')
parser.add_argument('--bilevel_batch_count',default=60,type=int,help='number of batches before switching the training modes')
parser.add_argument('--barely_train',action='store_true',help='not a real run')
parser.add_argument('--resume', '-r',action='store_true',help='resume from checkpoint')
parser.add_argument('--gate',type=GateType,default=GateType.UNCERTAINTY,choices=GateType)
parser.add_argument('--drop-path',type=float,default=0.1,metavar='PCT',help='Drop path rate (default: None)')
parser.add_argument('--gate_objective', type=GateObjective, default=GateObjective.CrossEntropy, choices=GateObjective)
parser.add_argument('--transfer-ratio',type=float,default=0.01, help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--proj_dim',default=32,help='Target dimension of random projection for ReLU codes')
parser.add_argument('--num_proj',default=16,help='Target number of random projection for ReLU codes')
parser.add_argument('--use_mlflow',default=True, help='Store the run with mlflow')
parser.add_argument('--classifier_loss', type=LossContributionMode, default=LossContributionMode.BOOSTED, choices=LossContributionMode)
parser.add_argument('--early_exit_warmup', default=True)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--split_ratio', default=0.8, type=float)
parser.add_argument('--eval', action='store_true', help='Enabling distributed evaluation')

# 解析命令行参数
args = parser.parse_args()

# 设置随机种子
seed = args.seed
print("seed:",seed)
torch.manual_seed(seed)
np.random.seed(seed)

# 初始化 MLflow 实验日志
if args.use_mlflow:
    name = "_".join([str(a) for a in [args.ce_ic_tradeoff, args.classifier_loss]])
    cfg = vars(args)
    if args.barely_train:
        experiment_name = 'test_run'    
    else:
        experiment_name = now.strftime("%m-%d-%Y")
    setup_mlflow(name, cfg, experiment_name=experiment_name)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
path_project = get_path_to_project_root()
model = args.arch


#####加载遥感数据集
# 获取数据集配置
dataset_config = get_dataset_config(args.data_set)

if dataset_config is None:
    raise ValueError(f"Unsupported dataset: {args.data_set}. Supported datasets: AID, NaSC, NWPU, PatternNet, UCM")

# 验证数据路径
if args.data_path is None:
    if dataset_config.get('default_data_path'):
        args.data_path = dataset_config['default_data_path']
    else:
        raise ValueError(f"--data-path must be specified for dataset {args.data_set}")

# 应用默认输入尺寸（如果使用默认值）
if args.input_size == 224:  # 默认值
    args.input_size = dataset_config['default_input_size']

# 根据配置决定使用哪种数据加载方式
if dataset_config['use_random_split']:
    # NaSC使用随机划分
    IMG_SIZE = args.input_size
    dataset_train, dataset_val, NUM_CLASSES = build_dataset_new(
        args=args, seed=args.seed, split_ratio=args.split_ratio
    )
else:
    # AID, NWPU, PatternNet, UCM使用固定划分
    IMG_SIZE = args.input_size
    dataset_train, NUM_CLASSES = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=args.batch,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=int(1.5 * args.batch),
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False
)



# 加载模型权重
checkpoint = None
if args.checkpoint_path is not None:
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(device))
elif not args.eval:
    # 训练模式需要checkpoint，但eval模式可能不需要（如果从头开始）
    print("Warning: No checkpoint path specified. Model will start from scratch.")

print(f'learning rate:{args.lr}, weight decay: {args.wd}\n')
print('==> Building model..')


# 构建模型
model = GFNet_xs_dynn(
            img_size=IMG_SIZE, num_classes=NUM_CLASSES,
            patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )


# 设置早退机制相关参数
transformer_layer_gating = [3, 5, 7, 9]
args.G = len(transformer_layer_gating)

model.set_CE_IC_tradeoff(args.ce_ic_tradeoff)
model.set_intermediate_heads(transformer_layer_gating)
model.set_learnable_gates(transformer_layer_gating, direct_exit_prob_param=True)


# 模型 FLOPs 计算 
n_flops, n_params, n_flops_at_gates = measure_model_and_assign_cost_per_exit(model, IMG_SIZE, IMG_SIZE, num_classes=NUM_CLASSES)
mult_add_at_exits = (torch.tensor(n_flops_at_gates) / 1e6).tolist()
print(mult_add_at_exits)


# 模型参数初始化
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_without_ddp = model.module


# 加载权重并检查参数匹配
if checkpoint is not None:
    print('==> Resuming from checkpoint..')
    param_with_issues = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print("Missing keys:", param_with_issues.missing_keys)
    print("Unexpected keys:", param_with_issues.unexpected_keys)
else:
    print('==> Starting from scratch (no checkpoint provided)')
total_params = sum(p.numel() for p in model_without_ddp.parameters())
print(f"Total number of parameters: {total_params}")


# 冻结主干网络，仅训练分类头和 gate
unfrozen_modules = ['intermediate_heads', 'gates']
freeze_backbone_helper(model, unfrozen_modules)
parameters = model.parameters()
optimizer = optim.SGD(parameters,
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       eta_min=args.min_lr,
                                                       T_max=args.num_epoch)



# 开始
# │
# ├── args.eval == True?
# │   └── 是：推理模式
# │       └── 对每个 threshold 执行动态推理
# │           └── 输出 acc 和 GFLOPs
# │
# └── 否：训练模式
#     ├── 初始化 learning_helper
#     ├── 执行 warmup 阶段(训练主分类头，冻结所有中间分类头)
#     ├── 解冻中间分类头
#     ├── 进入主训练循环
#     │   ├── train_single_epoch
#     │   ├── evaluate
#     │   ├── test_layer
#     │   ├── set_from_validation
#     │   └── scheduler.step()
#     └── 结束


if args.eval:
    num = 1
    test_stats = test_layer(model, data_loader_val, num, device=device)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(len(thresholds)):
        sum_acc = 0
        exp_flops = 0
        for j in range(num):
            acc, temp_flops = perform_test(model, data_loader_val, threshold=thresholds[i], flops=mult_add_at_exits, device=device)
            sum_acc += acc
            exp_flops += temp_flops
        sum_acc = sum_acc / num
        exp_flops = exp_flops / num
        print(f"threshold:{thresholds[i]}-->expected_acc:{sum_acc}<=>expected_GFLOPs{exp_flops}G")
else:
    # args.max_warmup_epoch = 10
    best_acc = 0
    # # start with warm up for the first epoch
    learning_helper = LearningHelper(model, optimizer, args, device)

    warmup_epoch = dynamic_warmup(args, learning_helper, device, data_loader_train, data_loader_val, IMG_SIZE)

    # Unfreeze all classifiers after warmup
    print("Unfreezing classifiers after warmup")
    model.module.unfreeze_all_intermediate_classifiers()

    for epoch in range(warmup_epoch + 1, args.num_epoch):
        train_single_epoch(args, learning_helper, device, data_loader_train, epoch=epoch, training_phase="classifier", bilevel_batch_count=args.bilevel_batch_count)

        val_metrics_dict, latest_acc, _ = evaluate(best_acc, args, learning_helper, device, data_loader_val, epoch, mode='val', experiment_name=experiment_name)
        test_stats = test_layer(model, data_loader_val, epoch, device=device)
        
        if latest_acc > best_acc: 
        
            best_acc = latest_acc
    
        set_from_validation(learning_helper, val_metrics_dict)
        scheduler.step()


