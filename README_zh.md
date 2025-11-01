# GFNet-Dynn: 基于早退机制的高能效边缘智能算法

**GFNet-Dynn** 是一个结合了 [GFNet](https://github.com/raoyongming/GFNet) 和 [Dynn](https://github.com/networkslab/dynn) 架构优点的混合深度学习模型。本项目旨在通过引入**动态退出机制**（即“早退机制”），在保持高精度的同时，显著提升深度神经网络在边缘设备上的推理效率。

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/KaHimm/GFNet-Dynn.git
cd GFNet-Dynn
```

### 2. 安装依赖

请参考 GFNet 和 Dynn 的官方指南安装必要的 Python 依赖包。

### 3. 准备数据集

将支持的数据集放置在项目根目录下的 `data/` 文件夹中。

**支持的数据集**：

- `AID`
- `NWPU`
- `PatternNet`
- `UCM`
- `NaSC`

> 📁 **注意**：数据集的路径和结构配置在 `data_loader_help.py` 和 `dataset_config.py` 中定义，请确保与实际路径一致。

### 4. 模型训练

使用提供的示例脚本进行训练：

```bash
./example_usage.sh
```

或手动运行：

```bash
python main_dynn.py \
    --data-set NWPU \
    --data-path /path/to/NWPU \
    --checkpoint-path /path/to/pretrained/checkpoint.pth \
    --batch 64 \
    --num_epoch 200 \
    --ce_ic_tradeoff 0.75 \
    --lr 0.001 \
    --input-size 256
```

### 5. 模型评估

使用 `--eval`参数评估预训练模型：

```bash
python main_dynn.py \
    --data-set NWPU \
    --data-path /path/to/NWPU \
    --checkpoint-path /path/to/pretrained/checkpoint.pth \
    --eval \
    --input-size 256
```

------

## ⚙️ 主要参数说明

| 参数                | 说明                                                    | 默认值       |
| :------------------ | :------------------------------------------------------ | :----------- |
| `--data-set`        | 数据集名称 (`UCM`, `NaSC`, `NWPU`, `PatternNet`, `AID`) | 必需         |
| `--data-path`       | 数据集根路径                                            | 必需         |
| `--checkpoint-path` | 检查点保存路径（训练）或模型路径（评估）                | `None`       |
| `--input-size`      | 输入图像尺寸                                            | 数据集默认值 |
| `--batch`           | 批处理大小                                              | `64`         |
| `--num_epoch`       | 训练轮数                                                | `100`        |
| `--ce_ic_tradeoff`  | 分类误差与推理成本的权衡系数                            | `0.75`       |
| `--split-ratio`     | 训练/验证集划分比例（随机划分时有效）                   | `0.8`        |

------

## 🏗️ 模型训练流程

GFNet-Dynn 采用两阶段训练策略以优化早退性能：

### 阶段一：动态 Warmup

- 逐个训练每个中间分类器（exit）。
- 根据验证集准确率，冻结表现最佳的 exit 分类器。
- 若某个 exit 连续 `patience` 轮未提升，则保存其最优权重并冻结。
- 若所有 exit 均收敛，则提前终止该阶段。

### 阶段二：完整模型训练

- 一起训练主干网络（backbone）和中间分类器（exit），以进一步提升整体性能。
- 最终模型支持在推理时根据置信度动态选择退出点。

------

## 📁 检查点文件管理

训练过程中的模型检查点（checkpoint）默认保存在 `checkpoint/` 目录下。

### 目录结构

```
├── checkpoint/
	    └── checkpoint_{dataset}_{ce_ic_tradeoff}_confEE/       # 不同实验配置的训练结果
├── checkpoint_warmup/                                          # Warmup 阶段的检查点
└── checkpoint_result/                                          # 人工筛选保存的最终模型
```

------

## 🛠️ 数据集配置

所有数据集的配置信息统一管理在 `dataset_config.py` 文件中。

```python
DATASET_CONFIG = {
    'UCM': {
        'num_classes': 21,
        'default_input_size': 256,
        'train_folder': 'train',
        'val_folder': 'val',
        'use_random_split': False,
        'default_data_path': './data/UCM'
    },
    # ... 其他数据集
}
```

> 📌 **提示**：如需添加新数据集或修改现有配置，请编辑此文件。



## 📜 参考资料

此项目借鉴了以下仓库的内容：

- [GFNet](https://github.com/raoyongming/GFNet)
- [Dynn](https://github.com/networkslab/dynn)

请参阅这些仓库以获取有关底层架构和方法的更多详细信息。