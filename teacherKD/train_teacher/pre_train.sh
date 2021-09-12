#!/usr/bin/env python
python -u pre_train.py \
    --arch resnet \
    --depth 18 \
    --dataset cifar10 \
    --epoch 160 \
    --optmzr sgd \
    --lr 0.01 \
    --lr-scheduler default \
    --save_root ckpt/baseline \
#    --model ckpt/resnet50-19c8e357.pth \

