#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import sys
import argparse
import caffe
import cv2
import numpy as np
import os
from os.path import exists, join, split, splitext
from collections import defaultdict, Counter

import dill
import network
import map_util
from shutil import copyfile
import logging
import datetime
import pandas as pd
import pickle as pkl
from plot_roc_curve import plot_multiple_roc_curves, calculate_AUC

log = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s')
log.setLevel(logging.INFO)

__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'

#TODO: fix it for case with more than four classes.
CLASS_COLOR_DICT = defaultdict(lambda: (128, 128, 128),
                               {0: (0, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 4: (0, 192, 192)})

def read_array(filename):
    with open(filename, 'rb') as fp:
        type_code = np.fromstring(fp.read(4), dtype=np.int32)
        shape_size = np.fromstring(fp.read(4), dtype=np.int32)
        shape = np.fromstring(fp.read(4 * shape_size), dtype=np.int32)
        if type_code == cv2.CV_32F:
            dtype = np.float32
        if type_code == cv2.CV_64F:
            dtype = np.float64
        return np.fromstring(fp.read(), dtype=dtype).reshape(shape)


def write_array(filename, array):
    with open(filename, 'wb') as fp:
        if array.dtype == np.float32:
            typecode = cv2.CV_32F
        elif array.dtype == np.float64:
            typecode = cv2.CV_64F
        else:
            raise ValueError("type is not supported")
        fp.write(np.array(typecode, dtype=np.int32).tostring())
        fp.write(np.array(len(array.shape), dtype=np.int32).tostring())
        fp.write(np.array(array.shape, dtype=np.int32).tostring())
        fp.write(array.tostring())


def make_frontend_vgg(options):
    deploy_net = caffe.NetSpec()
    deploy_net.data = network.make_input_data(options.input_size)
    last, final_name = network.build_frontend_vgg(
        deploy_net, deploy_net.data, options.classes)
    if options.up:
        deploy_net.upsample = network.make_upsample(last, options.classes)
        last = deploy_net.upsample
    deploy_net.prob = network.make_prob(last)
    deploy_net = deploy_net.to_proto()
    return deploy_net, final_name


def make_context(options):
    deploy_net = caffe.NetSpec()
    deploy_net.data = network.make_input_data(
        options.input_size, options.classes)
    last, final_name = network.build_context(
        deploy_net, deploy_net.data, options.classes, options.layers)
    if options.up:
        deploy_net.upsample = network.make_upsample(last, options.classes)
        last = deploy_net.upsample
    deploy_net.prob = network.make_prob(last)
    deploy_net = deploy_net.to_proto()
    return deploy_net, final_name


def make_joint(options):
    deploy_net = caffe.NetSpec()
    deploy_net.data = network.make_input_data(options.input_size)
    last = network.build_frontend_vgg(
        deploy_net, deploy_net.data, options.classes)[0]
    last, final_name = network.build_context(
        deploy_net, last, options.classes, options.layers)
    if options.up:
        deploy_net.upsample = network.make_upsample(last, options.classes)
        last = deploy_net.upsample
    deploy_net.prob = network.make_prob(last)
    deploy_net = deploy_net.to_proto()
    return deploy_net, final_name


def make_deploy(options):
    return globals()['make_' + options.model](options)


def write_img_gt_pred(result_dir, image_path, image_id, gt_mask, prediction, type):
    result_dir = os.path.join(result_dir, type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def _output_file_name(postfix):
        return image_id + '_' + postfix + image_path.split('/')[-1][-4:]

    # target image
    copyfile(image_path, os.path.join(result_dir, _output_file_name('im')))

    # ground truth label image
    gt_mask = gt_mask[:, :, 0]
    gt_mask_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    gt_mask_rgb[gt_mask == 0] = (255, 255, 255)
    for l in CLASS_COLOR_DICT.keys():
        if l != 0:
            gt_mask_rgb[gt_mask == l] = CLASS_COLOR_DICT[l]
    cv2.imwrite(os.path.join(result_dir, _output_file_name('lb')), gt_mask_rgb)

    # make prediction label image in rgb and write
    prediction_rgb = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for label in CLASS_COLOR_DICT.keys():
        prediction_rgb[prediction == label] = CLASS_COLOR_DICT[label]
    cv2.imwrite(os.path.join(result_dir, _output_file_name('lp')), prediction_rgb)


def test_image(options):
    options.feat_dir = join(options.feat_dir, options.feat_layer_name)
    if not exists(options.feat_dir):
        os.makedirs(options.feat_dir)

    label_margin = 186

    if options.up:
        zoom = 1
    else:
        zoom = 8

    if options.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(options.gpu)
        print('Using GPU ', options.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')

    mean_pixel = np.array(options.mean, dtype=np.float32)
    net = caffe.Net(options.deploy_net, options.weights, caffe.TEST)

    image_paths = [line.strip() for line in open(options.image_list, 'r')]
    image_names = [split(p)[1] for p in image_paths]
    input_dims = list(net.blobs['data'].shape)

    label_paths = []
    if options.label_list is not None:
        label_paths = [line.strip() for line in open(options.label_list, 'r')]

    image_ids = [split(path)[1][:-4] for path in image_paths]
    if options.image_id_list is not None:
        image_ids = [line.strip() for line in open(options.image_id_list, 'r')]

    item_ids = image_ids
    if options.item_id_list is not None:
        item_ids = [line.strip() for line in open(options.item_id_list, 'r')]

    assert input_dims[0] == 1
    batch_size, num_channels, input_height, input_width = input_dims
    print('Input size:', input_dims)
    caffe_in = np.zeros(input_dims, dtype=np.float32)

    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin

    feat_list = []

    gt_list = []
    tp = defaultdict(float)
    fp = defaultdict(float)
    fn = defaultdict(float)

    class_names = [name for name in options.class_names.split(',')]
    assert len(class_names) == options.classes, 'class names and number of classes do not match.'

    item_max_prob_dict = defaultdict(lambda: defaultdict(float))
    item_gt_label_dict = defaultdict(lambda: [0]*len(class_names))


    for i in range(len(image_names)):
        print('Predicting', image_ids[i])
        image = cv2.imread(image_paths[i]).astype(np.float32) - mean_pixel
        image_size = image.shape
        print('Image size:', image_size)
        image = cv2.copyMakeBorder(image, label_margin, label_margin,
                                   label_margin, label_margin,
                                   cv2.BORDER_REFLECT_101)
        num_tiles_h = image_size[0] // output_height + \
                      (1 if image_size[0] % output_height else 0)
        num_tiles_w = image_size[1] // output_width + \
                      (1 if image_size[1] % output_width else 0)
        prediction = []
        feat = []
        for h in range(num_tiles_h):
            col_prediction = []
            col_feat = []
            for w in range(num_tiles_w):
                offset = [output_height * h,
                          output_width * w]
                tile = image[offset[0]:offset[0] + input_height,
                       offset[1]:offset[1] + input_width, :]
                margin = [0, input_height - tile.shape[0],
                          0, input_width - tile.shape[1]]
                tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                          margin[2], margin[3],
                                          cv2.BORDER_REFLECT_101)
                caffe_in[0] = tile.transpose([2, 0, 1])
                blobs = []
                if options.bin:
                    blobs = [options.feat_layer_name]
                out = net.forward_all(blobs=blobs, **{net.inputs[0]: caffe_in})
                prob = out['prob'][0]
                if options.bin:
                    col_feat.append(out[options.feat_layer_name][0])
                col_prediction.append(prob)
            col_prediction = np.concatenate(col_prediction, axis=2)
            if options.bin:
                col_feat = np.concatenate(col_feat, axis=2)
                feat.append(col_feat)
            prediction.append(col_prediction)
        prob = np.concatenate(prediction, axis=1)
        if options.bin:
            feat = np.concatenate(feat, axis=1)

        if zoom > 1:
            zoom_prob = map_util.interp_map(
                prob, zoom, image_size[1], image_size[0])
        else:
            zoom_prob = prob[:, :image_size[0], :image_size[1]]

        # make prediction
        prediction = np.argmax(zoom_prob.transpose([1, 2, 0]), axis=2)

        # save predicted score dictionary
        item_id = item_ids[i]
        item_max_prob_dict[item_id] = [max(item_max_prob_dict[item_id][c], np.max(zoom_prob[c]))
                                       for c in range(zoom_prob.shape[0])]

        if options.bin:
            out_path = join(options.feat_dir,
                            splitext(image_names[i])[0] + '.bin')
            print('Writing', out_path)
            write_array(out_path, feat.astype(np.float32))
            feat_list.append(out_path)

        # prediction and ground truth set
        pred_set = (set(prediction.flatten()) - {0})
        gt_mask = cv2.imread(label_paths[i])
        gt_set = set(gt_mask.flatten()) - {255}

        # # count tp, fp and ground truth for calculation of false positive rate and miss rate
        # # and write result images
        print('{}/{} patch: {} '.format(i, len(image_names), image_names[i]))
        print('ground truth : {}'.format(gt_set))
        print('prediction: {}'.format(pred_set))
        gt_list += list(gt_set.union({0}))
        for f_pr in pred_set:
            if f_pr in gt_set:
                tp[f_pr] += 1.0
                write_img_gt_pred(options.result_dir, image_paths[i], image_ids[i], gt_mask, prediction, 'tp%d' % f_pr)
            else:
                fp[f_pr] += 1.0
                write_img_gt_pred(options.result_dir, image_paths[i], image_ids[i], gt_mask, prediction, 'fp%d' % f_pr)

        for f_gt in gt_set:
            if not (f_gt in pred_set):
                fn[f_gt] += 1.0
                write_img_gt_pred(options.result_dir, image_paths[i], image_ids[i], gt_mask, prediction, 'fn%d' % f_gt)

        item_gt_label_dict[item_id] = np.max((item_gt_label_dict[item_id],
                                          [1 if c in gt_set else 0 for c in range(len(class_names))])
                                          , axis=0)


    gt_count = Counter(gt_list)
    result_fnr_fpr = dict()
    for l in gt_count.keys():
        if l != 0:
            print('----------------------------image by image evaluation-------------------------------------------')
            print('false negative rate for %d : %.3f' % (l, (1 - (tp[l] / gt_count[l]))))
            result_fnr_fpr['fnr_%s' % class_names[l]] = (1 - (tp[l] / gt_count[l]))
            if gt_count[l] != gt_count[0]:
                print('false positive rate for %d : %.3f' % (l, (fp[l] / (gt_count[0] - gt_count[l]))))
                result_fnr_fpr['fpr_%s' % class_names[l]] = fp[l] / (gt_count[0] - gt_count[l])
            else:
                print('cannot calculate FPR: there is no normal data (only faults) for %d' % l)
                result_fnr_fpr['fpr_%s' % class_names[l]] = -1

    print('\nwriting result...\n')

    plot_multiple_roc_curves(item_gt_label_dict, item_max_prob_dict, class_names, options.result_dir)
    AUC_dict = calculate_AUC(item_gt_label_dict, item_max_prob_dict, [1, 2, 3, 4])

    # write experiment result
    with open(os.path.join(options.result_dir, 'item_max_prob_dict.pkl'), 'wb') as f:
        pkl.dump(item_max_prob_dict, f)

    with open(os.path.join(options.result_dir, 'item_gt_label_dict.pkl'), 'wb') as f:
        pkl.dump(item_gt_label_dict, f)

    result_auc = {'auc_%s' % class_names[c]: AUC_dict[c] for c in [1, 2, 3, 4]}
    result_tp = {'tp_%s' % class_names[c]: tp[c] for c in [1, 2, 3, 4]}
    result_fp = {'fp_%s' % class_names[c]: fp[c] for c in [1, 2, 3, 4]}
    result_fn = {'fn_%s' % class_names[c]: fn[c] for c in [1, 2, 3, 4]}

    wk_dir = '/'.join(options.weights.split('/')[:-2])
    iter = options.weights.split('_')[-1].split('.')[0]
    result_wkdir_iter = {'wkdir': wk_dir, 'iter': iter}

    result_merged = dict()
    result_merged['date'] = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    for d in [result_wkdir_iter, result_auc, result_fnr_fpr, result_tp, result_fp, result_fn]:
        for k, v in d.iteritems():
            result_merged[k] = v

    columns = ['date', 'wkdir', 'iter']
    for prefix in ['auc', 'fnr', 'fpr', 'tp', 'fn', 'fp']:
        columns += ['%s_%s' % (prefix, class_names[c]) for c in range(1, 5)]

    integrated_result_csv_path = options.integrated_result_csv_path
    if integrated_result_csv_path is None:
        print('warning: integrated result csv file path does not exist.')
    else:
        if not os.path.exists(integrated_result_csv_path):
            result_csv = pd.DataFrame(columns=columns)
        else:
            result_csv = pd.read_csv(integrated_result_csv_path)

        result_csv = result_csv.append(pd.Series(result_merged), ignore_index=True)
        result_csv = result_csv[columns]
        result_csv.to_csv(integrated_result_csv_path, index=False)

    print('================================')
    print('All results are generated.')
    print('================================')

def test_bin(options):
    label_margin = 0
    input_zoom = 8
    pad = 0
    if options.up:
        zoom = 1
    else:
        zoom = 8

    if options.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(options.gpu)
        print('Using GPU ', options.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')

    net = caffe.Net(options.deploy_net, options.weights, caffe.TEST)

    image_paths = [line.strip() for line in open(options.image_list, 'r')]
    bin_paths = [line.strip() for line in open(options.bin_list, 'r')]
    names = [splitext(split(p)[1])[0] for p in bin_paths]

    assert len(image_paths) == len(bin_paths)

    input_dims = net.blobs['data'].shape
    assert input_dims[0] == 1
    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)

    bin_test_image = read_array(bin_paths[0])
    bin_test_image_shape = bin_test_image.shape
    assert bin_test_image_shape[1] <= input_height and \
           bin_test_image_shape[2] <= input_width, \
        'input_size should be greater than bin image size {} x {}'.format(
            bin_test_image_shape[1], bin_test_image_shape[2])

    result_list = []

    for i in range(len(image_paths)):
        print('Predicting', bin_paths[i])
        image = cv2.imread(image_paths[i])
        image_size = image.shape
        if input_zoom != 1:
            image_rows = image_size[0] // input_zoom + \
                         (1 if image_size[0] % input_zoom != 0 else 0)
            image_cols = image_size[1] // input_zoom + \
                         (1 if image_size[1] % input_zoom != 0 else 0)
        else:
            image_rows = image_size[0]
            image_cols = image_size[1]
        image_bin = read_array(bin_paths[i])
        image_bin = image_bin[:, :image_rows, :image_cols]

        top = label_margin
        bottom = input_height - top - image_rows
        left = label_margin
        right = input_width - left - image_cols

        for j in range(num_channels):
            if pad == 1:
                caffe_in[0][j] = cv2.copyMakeBorder(
                    image_bin[j], top, bottom, left, right,
                    cv2.BORDER_REFLECT_101)
            elif pad == 0:
                caffe_in[0][j] = cv2.copyMakeBorder(
                    image_bin[j], top, bottom, left, right,
                    cv2.BORDER_CONSTANT)
        out = net.forward_all(**{net.inputs[0]: caffe_in})
        prob = out['prob'][0]
        if zoom > 1:
            prob = map_util.interp_map(prob, zoom, image_size[1], image_size[0])
        else:
            prob = prob[:, :image_size[0], :image_size[1]]
        prediction = np.argmax(prob.transpose([1, 2, 0]), axis=2)
        out_path = join(options.result_dir, names[i] + '.png')
        print('Writing', out_path)
        cv2.imwrite(out_path, prediction)
        result_list.append(out_path)

    print('================================')
    print('All results are generated.')
    print('================================')

    result_list_path = join(options.result_dir, 'results.txt')
    print('Writing', result_list_path)
    with open(result_list_path, 'w') as fp:
        fp.write('\n'.join(result_list))


def test(options):
    if options.model == 'context':
        test_bin(options)
    else:
        test_image(options)


def process_options(options):
    assert exists(options.image_list), options.image_list + ' does not exist'
    assert exists(options.weights), options.weights + ' does not exist'
    assert options.model != 'context' or exists(options.bin_list), \
        options.bin_list + ' does not exist'

    if options.model == 'frontend':
        options.model += '_vgg'

    work_dir = options.work_dir
    model = options.model
    options.deploy_net = join(work_dir, model + '_deploy.txt')
    iter = options.weights.split('_')[-1].split('.')[0]
    datetime_now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    options.result_dir = join(work_dir, 'results', options.sub_dir, '_'.join([model, iter, datetime_now]))
    options.feat_dir = join(work_dir, 'bin', options.sub_dir, model)

    if options.input_size is None:
        options.input_size = [80, 80] if options.model == 'context' \
            else [900, 900]
    elif len(options.input_size) == 1:
        options.input_size.append(options.input_size[0])

    if not exists(work_dir):
        print('Creating working directory', work_dir)
        os.makedirs(work_dir)
    if not exists(options.result_dir):
        print('Creating', options.result_dir)
        os.makedirs(options.result_dir)
    if options.bin and not exists(options.feat_dir):
        print('Creating', options.feat_dir)
        os.makedirs(options.feat_dir)

    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?',
                        choices=['frontend', 'context', 'joint'])
    parser.add_argument('--work_dir', default='training/',
                        help='Working dir for training.')
    parser.add_argument('--sub_dir', default='',
                        help='Subdirectory to store the model testing results. '
                             'For example, if it is set to "val", the testing '
                             'results will be saved in <work_dir>/results/val/ '
                             'folder. By default, the results are saved in '
                             '<work_dir>/results/ directly.')
    parser.add_argument('--image_list', required=True,
                        help='List of images to test on. This is required '
                             'for context module to deal with variable image '
                             'size.')
    parser.add_argument('--label_list', required=False,
                        default=None, help='List of Label of test images.')
    parser.add_argument('--image_id_list', required=False,
                        default=None, help='List of id of test images.')
    parser.add_argument('--item_id_list', required=False,
                        default=None, help='List of item id of test images. '
                                           'It is used for calculating evaluation metrics')
    parser.add_argument('--bin_list', help='The input for context module')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--bin', action='store_true',
                        help='Turn on to output the features of a '
                             'layer. It can be useful to generate input for '
                             'context module.')
    parser.add_argument('--feat_layer_name', default=None,
                        help='Extract the response maps from this layer. '
                             'It is usually the penultimate layer. '
                             'Usually, default is good.')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52], type=float,
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--input_size', nargs='*', type=int,
                        help='The input image size for deploy network.')
    parser.add_argument('--classes', type=int, required=True,
                        help='Number of categories in the data')
    parser.add_argument('--class_names', required=False,
                        default=None, help='name of categories. it should be matched with "classes" and '
                                           'written with comma separated format.\n '
                                           'ex) \'cat1,cat2,cat3\' for 3 categories')
    parser.add_argument('--up', action='store_true',
                        help='If true, upsample the final feature map '
                             'before calculating the loss or accuracy')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU for testing. If it is less than 0, '
                             'CPU is used instead.')
    parser.add_argument('--layers', type=int, default=8,
                        help='Used for training context module.\n'
                             'Number of layers in the context module.')
    parser.add_argument('--integrated_result_csv_path', required=False,
                        default=None, help='path of csv file for result integrated with other experiment results')

    options = process_options(parser.parse_args())
    deploy_net, feat_name = make_deploy(options)
    if options.feat_layer_name is None:
        options.feat_layer_name = feat_name
    print('Writing', options.deploy_net)
    with open(options.deploy_net, 'w') as fp:
        fp.write(str(deploy_net))
    test(options)


if __name__ == '__main__':
    main()
