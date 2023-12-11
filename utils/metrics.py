from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd


def scores(y_true, y_predict):
    '''
    计算田路分割任务中的各种评价指标
    原数据集中road=0,field=1
    :param y_true: 真实标签
    :param y_predict: 预测标签
    :return: 田路分割任务中的各种评价指标字典
    '''
    # 以field为正类，计算指标
    field_r = recall_score(y_true, y_predict)
    field_p = precision_score(y_true, y_predict)
    field_f1 = f1_score(y_true, y_predict)
    # 以road为正类，计算指标
    road_r = recall_score(y_true, y_predict, pos_label=0)
    road_p = precision_score(y_true, y_predict, pos_label=0)
    road_f1 = f1_score(y_true, y_predict, pos_label=0)
    # 计算总体指标
    pre=(field_p+road_p)/2
    rec=(field_r+road_r)/2
    acc = accuracy_score(y_true, y_predict)
    macro_f1 = f1_score(y_true, y_predict, average='macro')
    return {
        'precision':pre,
        'recall':rec,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'field_recall': field_r,
        'field_precision': field_p,
        'field_f1': field_f1,
        'road_recall': road_r,
        'road_precision': road_p,
        'road_f1': road_f1,
    }
