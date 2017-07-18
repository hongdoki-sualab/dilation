#!/usr/bin/env bash
DATA_DIR=../dataset/ISM/dilation/txt/target_defect_raw
WK_DIR=../output/ISM/dilation/training
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:caffe-dilation/build_master/lib PYTHONPATH=caffe-dilation/build_master/python \

for ITER in 103
do
    python test.py frontend \
        --work_dir $WK_DIR \
        --mean 94.22 94.22 94.22 \
        --image_list $DATA_DIR/test_image_list.txt \
        --label_list $DATA_DIR/test_label_list.txt \
        --image_id_list $DATA_DIR/test_id_list.txt \
        --item_id_list $DATA_DIR/test_itemid_list.txt \
        --weights $WK_DIR/snapshots/frontend_vgg_iter_$ITER.caffemodel \
        --classes 5 \
        --class_names 0normal,1scratch,2blackspot,3overflow,4cutoff \
        --integrated_result_csv_path ../output/ISM/dilation/result.csv
done

