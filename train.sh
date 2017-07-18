#!/usr/bin/env bash

WK_DIR=../output/ISM/dilation/training
DATA_DIR=../dataset/ISM/dilation/txt/source_only
mkdir -p $WK_DIR/log

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:caffe-dilation/build_master_release/lib \
PYTHONPATH=caffe-dilation/build_master_release/python \
  python train.py frontend \
  --work_dir $WK_DIR \
  --train_image $DATA_DIR/training_image_list.txt \
  --train_label $DATA_DIR/training_label_list.txt \
  --test_image $DATA_DIR/test_image_list.txt \
  --test_label $DATA_DIR/test_label_list.txt \
  --classes 5 \
  --weights pretrained/vgg_conv.caffemodel \
  --caffe caffe-dilation/build_master/tools/caffe \
  --mean 101.89 101.89 101.89 \
  --train_batch 12 \
  --test_batch 4 \
  --crop_size 380 \
  --lr 0.001 \
  --momentum 0.9 \
&> $WK_DIR/log/$(date +%Y-%m-%d-%H:%M:%S).log
#  --snapshot_to_resume $WK_DIR/snapshots/frontend_vgg_iter_.solverstate \
#  --solver_to_resume $WK_DIR/frontend_vgg_solver.txt \

