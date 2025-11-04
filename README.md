# Lightweight Remote Sensing Scene Classification on Edge Devices via Knowledge Distillation and Early-exit

**GFNet-Dynn** is a hybrid deep learning model that combines the advantages of the [GFNet](https://github.com/raoyongming/GFNet) and [Dynn](https://github.com/networkslab/dynn) architectures. This project aims to significantly enhance the inference efficiency of deep neural networks on edge devices while maintaining high accuracy by introducing a **dynamic exit mechanism** (i.e., "early exit mechanism").

## 🚀 Quick Start

### 1. Clone the Repository

### 2. Install Dependencies

Please refer to the official guides of GFNet and Dynn to install the necessary Python dependencies.

### 3. Prepare the Dataset

Place the supported datasets in the `data/` folder at the project root directory.

**Supported Datasets**:

- `AID`
- `NWPU`
- `PatternNet`
- `UCM`
- `NaSC`

> 📁 **Note**: The paths and structures of the datasets are defined in `data_loader_help.py` and `dataset_config.py`. Please ensure they match the actual paths.

### 4. Model Training

Use the provided example script for training:

```bash
bash

./example_usage.sh
```

Or run it manually:

```bash
bashpython main_dynn.py \
    --data-set NWPU \
    --data-path /path/to/NWPU \
    --checkpoint-path /path/to/pretrained/checkpoint.pth \
    --batch 64 \
    --num_epoch 200 \
    --ce_ic_tradeoff 0.75 \
    --lr 0.001 \
    --input-size 256
```

### 5. Model Evaluation

Evaluate a pre-trained model using the `--eval` parameter:

```bash
bashpython main_dynn.py \
    --data-set NWPU \
    --data-path /path/to/NWPU \
    --checkpoint-path /path/to/pretrained/checkpoint.pth \
    --eval \
    --input-size 256
```

------

## ⚙️ Main Parameter Descriptions

| Parameter           | Description                                                  | Default Value         |
| :------------------ | :----------------------------------------------------------- | :-------------------- |
| `--data-set`        | Dataset name (`UCM`, `NaSC`, `NWPU`, `PatternNet`, `AID`)    | Required              |
| `--data-path`       | Root path of the dataset                                     | Required              |
| `--checkpoint-path` | Checkpoint save path (for training) or model path (for evaluation) | `None`                |
| `--input-size`      | Input image size                                             | Dataset default value |
| `--batch`           | Batch size                                                   | `64`                  |
| `--num_epoch`       | Number of training epochs                                    | `100`                 |
| `--ce_ic_tradeoff`  | Trade-off coefficient between classification error and inference cost | `0.75`                |
| `--split-ratio`     | Training/validation set split ratio (valid for random split) | `0.8`                 |

------

## 🏗️ Model Training Process

GFNet-Dynn adopts a two-stage training strategy to optimize early exit performance:

### Stage One: Dynamic Warmup

- Train each intermediate classifier (exit) one by one.
- Freeze the exit classifier with the best performance based on the validation set accuracy.
- If an exit has not improved for `patience` consecutive rounds, save its optimal weights and freeze it.
- If all exits have converged, terminate this stage early.

### Stage Two: Full Model Training

- Train the backbone network and intermediate classifiers (exits) together to further enhance overall performance.
- The final model supports dynamically selecting the exit point during inference based on confidence.

------

## 📁 Checkpoint File Management

Model checkpoints during training are saved by default in the `checkpoint/` directory.

### Directory Structure

```
├── checkpoint/
	    └── checkpoint_{dataset}_{ce_ic_tradeoff}_confEE/       # Training results for different experimental configurations
├── checkpoint_warmup/                                          # Checkpoints from the warmup stage
└── checkpoint_result/                                          # Final models manually selected and saved
```

------

## 🛠️ Dataset Configuration

All dataset configuration information is uniformly managed in the `dataset_config.py` file.

```python
pythonDATASET_CONFIG = {
    'UCM': {
        'num_classes': 21,
        'default_input_size': 256,
        'train_folder': 'train',
        'val_folder': 'val',
        'use_random_split': False,
        'default_data_path': './data/UCM'
    },
    # ... Other datasets
}
```

> 📌 **Tip**: To add a new dataset or modify an existing configuration, please edit this file.

## 📜 References

This project draws on content from the following repositories:

- [GFNet](https://github.com/raoyongming/GFNet)
- [Dynn](https://github.com/networkslab/dynn)

Please refer to these repositories for more detailed information about the underlying architectures and methods.
