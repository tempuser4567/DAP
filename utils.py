# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import importlib
import zipfile
import json
from scipy.interpolate import interp1d
import datetime, pytz, hashlib, time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve
from eval_box.module_map import module_map, module_map_reverse, type_map, type_map_reverse, chinese_map # test
from ipdb import set_trace


from matplotlib import font_manager
my_font = font_manager.FontProperties(fname="./MSYH.TTC")


def accuracy_per_conf(gt, predict):
    return_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0-0.1, ..., 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0
    correct_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(gt)):
        if predict[i] > 0 and predict[i] <= 0.1:
            return_list[0] += 1
            if gt[i] == 1:
                correct_list[0] += 1
        elif predict[i] > 0.1 and predict[i] <= 0.2:
            return_list[1] += 1
            if gt[i] == 1:
                correct_list[1] += 1
        elif predict[i] > 0.2 and predict[i] <= 0.3:
            return_list[2] += 1
            if gt[i] == 1:
                correct_list[2] += 1
        elif predict[i] > 0.3 and predict[i] <= 0.4:
            return_list[3] += 1
            if gt[i] == 1:
                correct_list[3] += 1
        elif predict[i] > 0.4 and predict[i] <= 0.5:
            return_list[4] += 1
            if gt[i] == 1:
                correct_list[4] += 1
        elif predict[i] > 0.5 and predict[i] <= 0.6:
            return_list[5] += 1
            if gt[i] == 1:
                correct_list[5] += 1
        elif predict[i] > 0.6 and predict[i] <= 0.7:
            return_list[6] += 1
            if gt[i] == 1:
                correct_list[6] += 1
        elif predict[i] > 0.7 and predict[i] <= 0.8:
            return_list[7] += 1
            if gt[i] == 1:
                correct_list[7] += 1
        elif predict[i] > 0.8 and predict[i] <= 0.9:
            return_list[8] += 1
            if gt[i] == 1:
                correct_list[8] += 1
        elif predict[i] > 0.9 and predict[i] <= 1:
            return_list[9] += 1
            if gt[i] == 1:
                correct_list[9] += 1
    for i in range(len(return_list)):
        return_list[i] = correct_list[i] * 1.0 / (return_list[i] + 1e-8)

    return return_list


def conf_diff(conf):
    gt = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    diff = 0
    for i in range(10):
        diff += (gt[i] - conf[i]) ** 2
    return diff ** 0.5


def read_result(result_path, gt_path):
    gt = []
    predict = []
    with open(result_path, 'r') as f:
        _data = f.readlines()
        for data in _data:
            predict.append(float(data.split(',')[-1][:-1]))
    with open(gt_path, 'r') as f:
        _data = f.readlines()
        for data in _data:
            gt.append(float(data.split(',')[0]))
    return gt, predict

def read_result_vft(result, protocol):
    gt = []
    proposals = {}
    with open(result, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_path = line.split(',')[0]
            index = line.find('[')
            sliced = line[index:]
            pred_periods = eval(sliced)
            proposals[file_path] = pred_periods
    with open(protocol, 'r') as f:
        lines = f.readlines()
        for line in lines:
            gt.append(line.strip('\n'))
    return gt, proposals


def cal_acc(groud_truth, result, threshold):
    acc = 0
    for i in range(len(result)):
        if groud_truth[i] == 1 and result[i] >= threshold:
            acc += 1
        elif groud_truth[i] == 0 and result[i] < threshold:
            acc += 1
    acc = acc / len(groud_truth)

    return acc


def get_int(groud_truth, result, threshold):
    predict = []
    for i in range(len(result)):
        if result[i] >= threshold:
            predict.append(1)
        else:
            predict.append(0)

    return np.array(groud_truth, dtype=np.int64), np.array(predict, dtype=np.int64)


def compute_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer


def curve_convert(x, y):
    """
    :param x: 横坐标， 元素需要从小到大排序，可重复
    :param y: 对应纵坐标
    :return:
    """
    # new x
    x_ = np.linspace(0, 1, num=41)
    x_ = np.round(x_, 3)

    x1 = []
    y1 = []
    current_x = x[0]
    current_y = [y[0]]
    for i, xi in enumerate(x):
        if xi == current_x:
            current_y.append(y[i])
        else:
            x1.append(current_x)
            y1.append(sum(current_y) / len(current_y))
            current_x = xi
            current_y = [y[i]]
    x1.append(current_x)
    y1.append(sum(current_y) / len(current_y))
    # new y
    f = interp1d(x1, y1, kind='linear')
    y_ = f(x_)
    y_ = np.round(y_, 4)
    return x_.tolist(), y_.tolist()

def get_metrics(eval_box, taskroot, datasets, output):
    gts = []
    predicts = []
    Acc_list = []
    AUC_list = []
    EER_list = []
    Precision_list = []
    Recall_list = []
    F1_score_list = []
    conf_list = []
    dataset_list = []
    print(output, ' metrics...')
    for dataset in datasets:
        set_name = dataset.split('_')[0]
        dataset_list.append(set_name)
        result_path = os.path.join(taskroot, 'eval_outputs', output, dataset, 'eval_result.txt')
        if len(output.split('_')) > 3 and output.split('_')[1] == 'robustness':
            type = output.split('_')[-1]
            output0 = '_'.join(output.split('_')[0:-1])
            gt_path = os.path.join(eval_box, output0, type, 'groundtruth', '%s.txt' % dataset)
        else:
            gt_path = os.path.join(eval_box, output, 'groundtruth', '%s.txt' % dataset)
        gt, predict = read_result(result_path, gt_path)
        gts += gt
        predicts += predict
        fpr, tpr, thresholds = roc_curve(gt, predict, pos_label=1)
        EER_list.append(compute_eer(fpr, tpr, thresholds))

        threshold = 0.5

        gt_, predict_ = get_int(gt, predict, threshold)
        Acc_list.append(cal_acc(gt, predict, threshold))
        AUC_list.append(auc(fpr, tpr))
        Precision_list.append(precision_score(gt_, predict_))
        Recall_list.append(recall_score(gt_, predict_))
        F1_score_list.append(f1_score(gt_, predict_))
        conf = accuracy_per_conf(gt, predict)
        conf_list.append(conf_diff(conf))
    
    fpr, tpr, thresholds = roc_curve(gts, predicts, pos_label=1)

    roc_list = curve_convert(list(fpr), list(tpr))
    precisions, recalls, thresholds = precision_recall_curve(gts, predicts)
    pr_list = curve_convert(list(recalls), list(precisions))

    return Acc_list, AUC_list, EER_list, Precision_list, Recall_list, F1_score_list, conf_list, dataset_list, roc_list, pr_list

def get_topdown(taskroot, datasets, output):
    for dataset in datasets:
        result_path = os.path.join(taskroot, 'eval_outputs', output, dataset, 'eval_result.txt')
        total_dict = {}
        with open(result_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                total_dict[dataset + '_' + str(i)] = float(line.split(',')[-1][:-1])
    down_dict = dict(sorted(total_dict.items(),  key=lambda d: d[1], reverse=True)[:5])
    top_dict = dict(sorted(total_dict.items(),  key=lambda d: d[1], reverse=False)[:5])
    return top_dict, down_dict


def dict_append(dict, key, value):
    if key not in dict.keys():
        dict[key] = [value]
    else:
        dict[key].append(value)
    return dict

def iou_1d(proposal, target):
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is np.ndarray:
        proposal = torch.from_numpy(proposal)

    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union

if __name__ == '__main__':
    import json
