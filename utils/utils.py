#coding=utf-8
import os
import torch
import codecs
import copy
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from optim import Optim

def criterion(outputs, targets, avg=True):
    '''
    outputs: batch x classes, class probility
    targets: long tensor
    return loss list float tensor
    '''
    batch_size = len(targets)
    losses = -torch.log(outputs).gather(1, targets.view(-1,1))

    if avg:
        return torch.sum(losses) / batch_size
    else:
        return losses


def print_log(cls_index, filedir, epoch, x_data, lens, y_data, attns, preds,
               losses, dicts, flag=None, train_indices=None):
    '''
    epoch: int
    x_data: list of tensor, every with shape (batch size, seq len)
    lens: list of int
    y_data: list if tensor, every with shape (batch size, )
    attns: list of tensor,  shape (batch size, seq len)
    preds: list of tensor,  every with shape (batch size, classes)
    losses: list of tensor,  shape (batch size, )
    dicts: dicts, id:token
    flag: identity flag
    train_indices: index dict for reorder
    '''
    is_train = True if train_indices else False

    # every sentence has diff len
    x_data_new = []
    attns_new = []
    for _i in range(len(x_data)):
        x_data_new += [_j.squeeze(0) for _j in x_data[_i].split(1)]
        attns_new += [_j.squeeze(0) for _j in attns[_i].split(1)]

    y_data = torch.cat(y_data).squeeze(1)
    preds = torch.cat(preds).squeeze(1)
    preds_cls = preds.max(1)[1]

    losses = torch.cat(losses).squeeze(1)
    assert len(lens) == len(losses)

    assert len(lens) == len(x_data_new)
    assert len(lens) == len(y_data)
    assert len(lens) == len(attns_new)
    assert len(lens) == len(preds)
    
    res_data = []
    writed_data = []
    for i in range(len(lens)):

        if is_train:
            actual_i = train_indices[i]
        else:
            actual_i = i

        x = x_data_new[actual_i]
        prob = '/'.join([('%.2f'%_x) for _x in preds[actual_i]])
        writed_data.append('%d, gold: %d / pred: %d, %s, %6.4f' % (
                                                    i,
                                                    y_data[actual_i],
                                                    preds_cls[actual_i],
                                                    prob,
                                                    losses[actual_i]))
        res_data.append((i,
                        preds_cls[actual_i], 
                        y_data[actual_i], 
                        [_x for _x in preds[actual_i]],
                        losses[actual_i]))

        for j in range(lens[actual_i]):
            word = dicts[x[j]]
            if word == ',':
                word = '_'
            word = (word if len(word) <= 15 else word[:15]).ljust(15)
            a = '%.4f' % attns_new[actual_i][j]
            writed_data.append(word + ',' + a)
        writed_data.append('')

    directory = filedir + flag.split('_')[0] + '/'
    if not os.path.exists(directory):
            os.makedirs(directory)

    f = codecs.open(directory + 'cls' + str(cls_index) + '_e' + str(epoch)\
                + '_' + flag + '.csv', 'w', 'utf-8')
    f.write('\n'.join(writed_data))
    f.close()

    return res_data

def merge_res(res_data): # avg-max
    count = 0.0
    for i in range(len(res_data[0])):
        a = []
        for m_index in range(len(res_data)):
            a.append(res_data[m_index][i][3])

        pro = [sum(x) for x in zip(*a)]
        pred_index = np.argmax(pro)
        if pred_index == res_data[m_index][i][2]:
            count += 1

    return count / len(res_data[0])

def merge_res2(res_data): # min-max
    count = 0.0
    for i in range(len(res_data[0])):
        a = []
        for m_index in range(len(res_data)):
            a.append(res_data[m_index][i][3])

        a = np.array(a)
        pred_index = a.min(0).argmax()

        if pred_index == res_data[m_index][i][2]:
            count += 1

    return count / len(res_data[0])

def merge_res3(res_data): # max-max
    count = 0.0
    for i in range(len(res_data[0])):
        a = []
        for m_index in range(len(res_data)):
            a.append(res_data[m_index][i][3])

        a = np.array(a)
        pred_index = a.max(0).argmax()

        if pred_index == res_data[m_index][i][2]:
            count += 1

    return count / len(res_data[0])

def merge_res4(res_data): # 两个乘起来，归一化，取最大
    count = 0.0
    for i in range(len(res_data[0])):
        a = []
        for m_index in range(len(res_data)):
            a.append(res_data[m_index][i][3])

        prod = [np.prod(x) for x in zip(*a)]
        prod = prod / sum(prod)
        pred_index = prod.argmax()

        if pred_index == res_data[m_index][i][2]:
            count += 1

    return count / len(res_data[0])

def csoftmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def merge_res5(res_data, pred_chooses, k=None):

    pred_chooses = pred_chooses.numpy()
    N = len(pred_chooses[0]) # 总的分类数目

    def val(k):
        count, oracle_count = 0.0, 0.0
        for i in range(len(res_data[0])):
            a = []
            for m_index in range(len(res_data)):
                a.append(res_data[m_index][i][3])

            # cls picker acc
            prod = None
            if N <= 2: # 有点重复，其实可以不分2分类出来
                if pred_chooses[i][0] >= k:
                    prod = a[0]
                else:
                    prod = a[1]
            else:
                pred_choose = pred_chooses[i].argmax()
                prod = a[pred_choose]
            pred_index = np.array(prod).argmax()
            if pred_index == res_data[m_index][i][2]:
                count += 1

            # oracle acc
            # print res_data[0][i][4], res_data[1][i][4]
            c_losses = np.array([0]*N).astype('float32')
            for n in range(N):
                c_losses[n] = res_data[n][i][4]
            oracle_choose = c_losses.argmin()
            oracle_prod = a[oracle_choose]
            pred_index = np.array(oracle_prod).argmax()
            if pred_index == res_data[0][i][2]:
                oracle_count += 1

        acc = count / len(res_data[0])
        oracle_acc = oracle_count / len(res_data[0])

        return acc, oracle_acc

    best_th, best_acc, oracle_acc = 0, 0, 0

    if k:
        best_acc, oracle_acc = val(k)
    else:
        for th in range(100):
            threshold = float(th)/100
            last_acc, oracle_acc = val(threshold)
            if last_acc > best_acc:
                best_acc = last_acc
                best_th = threshold

    return best_acc, oracle_acc, best_th

