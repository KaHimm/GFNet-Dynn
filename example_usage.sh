#!/bin/bash
# 示例脚本：展示如何使用优化后的代码训练和评估模型

# ==========================================
# 示例1: 训练UCM数据集
# ==========================================
echo "示例1: 训练UCM数据集"
python main_dynn.py \
    --data-set UCM \
    --data-path "/workspace/data/UCM" \
    --checkpoint-path "/workspace/JEIDNN/checkpoint_result/checkpoint_UCM_GFNet-xs-dynn_(gate3579)/ckpt_epoch98_0.75_97.38.pth" \
    --batch 64 \
    --num_epoch 30 \
    --ce_ic_tradeoff 0.75 \
    --lr 1e-3 \
    --input-size 256 \
    --wd 1e-5

# ==========================================
# 示例2: 训练NaSC数据集（使用随机划分）
# ==========================================
echo "示例2: 训练NaSC数据集（使用随机划分）"
python main_dynn.py \
    --data-set NaSC \
    --data-path /path/to/NaSC \
    --checkpoint-path /path/to/pretrained/checkpoint.pth \
    --batch 64 \
    --num_epoch 200 \
    --ce_ic_tradeoff 0.75 \
    --lr 0.001 \
    --split-ratio 0.8 \
    --seed 42

# ==========================================
# 示例3: 评估模型
# ==========================================
echo "示例3: 评估模型"
python main_dynn.py \
    --data-set UCM \
    --data-path /path/to/UCM \
    --checkpoint-path /path/to/trained/checkpoint.pth \
    --eval \
    --batch 64

# ==========================================
# 示例4: 训练NWPU数据集
# ==========================================
echo "示例4: 训练NWPU数据集"
python main_dynn.py \
    --data-set NWPU \
    --data-path /path/to/NWPU \
    --checkpoint-path /path/to/pretrained/checkpoint.pth \
    --batch 64 \
    --num_epoch 200 \
    --ce_ic_tradeoff 0.75 \
    --lr 0.001 \
    --input-size 256

# ==========================================
# 示例5: 训练AID数据集（大尺寸输入）
# ==========================================
echo "示例5: 训练AID数据集（大尺寸输入）"
python main_dynn.py \
    --data-set AID \
    --data-path /path/to/AID \
    --checkpoint-path /path/to/pretrained/checkpoint.pth \
    --batch 32 \
    --num_epoch 200 \
    --ce_ic_tradeoff 0.75 \
    --lr 0.001 \
    --input-size 600

# ==========================================
# 示例6: 训练PatternNet数据集
# ==========================================
echo "示例6: 训练PatternNet数据集"
python main_dynn.py \
    --data-set PatternNet \
    --data-path /path/to/PatternNet \
    --checkpoint-path /path/to/pretrained/checkpoint.pth \
    --batch 64 \
    --num_epoch 200 \
    --ce_ic_tradeoff 0.75 \
    --lr 0.001 \
    --input-size 256

