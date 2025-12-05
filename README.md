# CL-ViME

## ⚙️ Configuration (`run_pretrain.sh`)

The shell script `run_pretrain.sh` configures and launches the pre-training:

```bash
#!/bin/bash

# Basic parameters
DATA_PATH=""  # Path to dataset
ARCH="vit_col_112"           # Model architecture (ViT with MMoE)
NAME="_test"                 # Experiment name suffix
BATCH_SIZE=2048              # Total batch size across GPUs
EPOCHS=300                   # Number of training epochs
WORKERS=8                    # Data loading workers per GPU
LR=3e-4                      # Learning rate
SAVE_PATH="./model"          # Checkpoint output directory

# MoCo-specific hyperparameters
M=0.99                       # Momentum for key encoder update
LAMDA=0.9                    # Weight for MMoE auxiliary loss
MLP_DIM=4096                 # MLP head hidden dimension (not used in script args but may be in model)
MOCO_DIM=250                 # Feature dimension of MoCo projection head
IMG_SIZE=60                  # Input image size (H = W = 60)

# Launch training on GPUs 0, 2, 3
CUDA_VISIBLE_DEVICES=0,2,3 python ./main_moco_mmoe.py \
    --data $DATA_PATH \
    --arch $ARCH \
    --name $NAME \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --workers $WORKERS \
    --lr $LR \
    --moco-m $M \
    --lamda $LAMDA \
    --img_size $IMG_SIZE \
    --save_path $SAVE_PATH
```

The shell script `run_lincls.sh` configures:

```bash
#!/bin/bash

# 第一个数据集配置
DATA_PATH=""


# 其他共享配置
DIST_URL="tcp://127.0.0.1:12386"
ARCH="vit_col_112"
PRETRAINED="/home/ubuntu/work/model/07-17-10MAWI1_5_tcp40_ut_60_vit_col_112/MAWI1_5_tcp40_ut_60_vit_col_112_0300.pth.tar"
BATCH_SIZE=1024
EPOCHS=200
WORKERS=4
LR=0.01
IMG_SIZE=60
NAME="01"

echo "正在运行数据集1: $DATA_PATH1"
CUDA_VISIBLE_DEVICES=2 python ./lincls_tvt.py \
    --data $DATA_PATH \
    --arch $ARCH \
    --pretrained $PRETRAINED \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --workers $WORKERS \
    --dist-url $DIST_URL \
    --lr $LR \
    --name $NAME \
    --img_size $IMG_SIZE
```
