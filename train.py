#coding=utf-8
from __future__ import division

import math
import time
import sys
import codecs
import copy
import argparse
import string
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from scipy.special import beta as Beta
from collections import OrderedDict
from tensorboardX import SummaryWriter

from utils.dataset import Dataset
from model.model import *
from utils.optim import *
from utils.utils import *
from utils.config import *


parser = argparse.ArgumentParser(description='train.py')

# dataset configuration
parser.add_argument('-config_file', type=str, required=True, # default='./trec/hyper-param.conf',
                    help='model configuration for a dataset, i.e. \'./trec/hyper-param-trec.conf\'')
parser.add_argument('-flag', type=str, default='',
                    help='unique flag ')

# GPU
parser.add_argument('-gpu', type=int, default=0,
                    help="Use CUDA on the listed devices, -1 for CPU")
parser.add_argument('-cudnn', action='store_false', default=True,
                    help="seed for reproductive.")
parser.add_argument('-seed', type=int, default=1234,
                    help="seed for reproductive.")

opt = parser.parse_args()
if os.path.exists(opt.config_file):
    config = Config()
    config.load_config(opt.config_file)
else:
    print '\n* NO CONFIG FILE *'
    sys.exit()

config.set('flag', opt.flag)
config.set('gpu', opt.gpu)
config.set('cudnn', opt.cudnn)
config.set('seed', opt.seed)
print config.lists()
del opt

if config['gpu'] != -1:
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.set_device(config['gpu'])
    torch.backends.cudnn.enabled=config['cudnn']
torch.manual_seed(config['seed'])


def init_cht0(layers, batch_size, hidden_size):
    h0 = Variable(torch.zeros(layers, batch_size, hidden_size))
    c0 = Variable(torch.zeros(layers, batch_size, hidden_size))

    if config['gpu'] != -1:
        h0 = h0.cuda()
        c0 = c0.cuda()
    return h0, c0

def init_context0(batch_size, input_size):
    context0 = Variable(torch.zeros(batch_size, input_size))

    if config['gpu'] != -1:
        context0 = context0.cuda()
    return context0

def eval(epoch, models, data, id2token_dict, flag=None):
    tokens_data = []
    lens_data = []
    labels_data = []

    h0s_np = []
    h1s_np = []
    pred0s_np = []
    pred1s_np = []
    ys_np = []
    xs_np = []
    les_np = []

    # tons of init
    prob_pred_datas = {}
    losses_datas = {}
    attns_datas = {}
    res_data = []
    total_losses = {}
    total_diff_losses, total_disc_losses, total_enpys = 0.0, 0.0, 0.0
    total_count = 0.0
    total_right_counts = {}
    report_ratios = {}
    total_ratios = {}
    pred_c_nums = {}

    total_pred_chooses = []

    for _i in range(config['hop'] + 1): # +1 for cls
        prob_pred_datas[_i] = []
        losses_datas[_i] = []
        attns_datas[_i] = []
        total_losses[_i], total_right_counts[_i] = 0, 0
        total_ratios[_i], pred_c_nums[_i] = 0, 0

    for m in models:
        models[m].eval()

    for i in range(len(data)):
        batch = data[i]
        x_data = batch[0] # including lens
        lens = x_data[1]
        y_data = batch[1]
        batch_size = len(y_data)

        tokens_data += [x_data[0].cpu().data]
        lens_data += x_data[1]
        labels_data += [y_data.cpu().data]

        ## C0 stuff
        # memory part
        memory = models['memory'](x_data)

        cs_losses = []
        cs_outputs = []
        cs_attns = []
        cs_hs = []
        for _i in range(config['hop']):
            if _i == 0:
                c_im1_context_t0s = init_context0(batch_size, config['mem_size'])
                c_im1_ch_t0s = init_cht0(config['layer_c'], batch_size, config['rnn_size_c'])
                # connector for first classifier
                c_im1_trigger_t0s = models['connector'](memory, lens)[0] # 0 for h

            c_i_outputs, c_i_attns, c_i_ch_t1s, c_i_context_t1s = models['C%d'%_i](memory,
                                                lens,
                                                c_im1_trigger_t0s,
                                                c_im1_context_t0s,
                                                c_im1_ch_t0s)
            c_i_losses = criterion(c_i_outputs, y_data, False)
            c_i_loss = torch.sum(c_i_losses) / batch_size

            cs_hs.append(c_i_ch_t1s[1][0])
            c_im1_context_t0s = c_i_context_t1s
            c_im1_ch_t0s = (c_i_ch_t1s[0], c_i_ch_t1s[1])
            c_im1_trigger_t0s = c_i_ch_t1s[1][0] # 0 for top layer

            cs_losses.append(c_i_losses)
            cs_outputs.append(c_i_outputs)
            cs_attns.append(c_i_attns)

        ## differentiated loss 
        # Beta distribution
        beta_value = (Beta(config['alpha'], config['beta'])).item()
        c0_ps = cs_outputs[0].gather(1, y_data.view(-1,1))
        reg = (1/beta_value) * torch.pow(c0_ps, config['alpha']-1)*torch.pow(1-c0_ps, config['beta']-1)
        c1_optim_losses = reg * cs_losses[1]

        c0_entropy = torch.sum(-cs_outputs[0]*torch.log(cs_outputs[0]), 1).unsqueeze(1)

        # this is inconsistent with the 'differentiated loss' in the papaer, since pytorch=0.2 has a bug, 
        # torch.mean(a+b)!=torch.mean(a)+torch.mean(b), so we add confidence pernalty directly
        other_loss = torch.mean(c1_optim_losses - config['enpy'] * c0_entropy)
        differentiated_loss = torch.mean(cs_losses[0] + c1_optim_losses)

        ## discriminator
        d_outputs = models['discriminator'](cs_hs)
        loss_dis = 1 - nn.Softmax()(torch.cat([c_losses/config['temp'] for c_losses in cs_losses], 1))
        d_loss = torch.mean(loss_dis*torch.pow(1-d_outputs, 0)*(-torch.log(d_outputs)))

        loss = torch.mean(cs_losses[0]) + other_loss + d_loss

        # discriminator outputs
        total_pred_chooses += [d_outputs.cpu().data]

        h0s_np.append(cs_hs[0].cpu().data.numpy())
        h1s_np.append(cs_hs[1].cpu().data.numpy())
        pred0s_np.append(cs_outputs[0].cpu().data.numpy())
        pred1s_np.append(cs_outputs[1].cpu().data.numpy())
        ys_np.append(y_data.cpu().data.numpy())
        xs_np += x_data[0].cpu().data.numpy().tolist()
        les_np += list(x_data[1])

        ## ton of print
        total_count += batch_size
        total_diff_losses += differentiated_loss.data[0] * batch_size
        total_disc_losses += d_loss.data[0] * batch_size
        total_enpys += torch.mean(c0_entropy).data[0] * batch_size
        for _i in range(config['hop']):
            # for print
            attns_datas[_i] += [cs_attns[_i].cpu().data]
            prob_pred_datas[_i] += [cs_outputs[_i].cpu().data]
            losses_datas[_i] += [cs_losses[_i].cpu().data]
            
            right = (y_data.squeeze(1)==cs_outputs[_i].max(1)[1]).sum().data[0]
            loss_float = cs_losses[_i].data[0]

            total_right_counts[_i] += right
            total_losses[_i] += loss_float*batch_size

            pred_c_nums[_i] += (d_outputs.data.max(1)[1] == _i).sum()
            gold_c_num = (loss_dis.data.max(1)[1] == _i).sum()
            total_ratios[_i] += gold_c_num

        # discriminator
        right = (loss_dis.max(1)[1]==d_outputs.max(1)[1]).sum().data[0]
        loss_float = d_loss.data[0]
        total_right_counts[_i + 1] += right
        total_losses[_i + 1] += loss_float*batch_size

    pred_chooses = torch.cat(total_pred_chooses)

    # averaging all this
    total_diff_loss = total_diff_losses / total_count
    total_disc_loss = total_disc_losses / total_count
    total_enpy = total_enpys / total_count
    for _i in range(config['hop'] + 1):
        total_losses[_i] /= total_count
        total_right_counts[_i] /= total_count
        if _i < config['hop']:
            total_ratios[_i] /= total_count
            pred_c_nums[_i] /= total_count

    for _i in range(config['hop']):
        res = print_log(_i, config['log'], epoch, tokens_data, lens_data, labels_data,
                                attns_datas[_i], prob_pred_datas[_i], losses_datas[_i], 
                                id2token_dict, flag)
        res_data.append(res)

    h0s_np = np.concatenate(h0s_np)
    h1s_np = np.concatenate(h1s_np)
    pred0s_np = np.concatenate(pred0s_np)
    pred1s_np = np.concatenate(pred1s_np)
    ys_np = np.concatenate(ys_np)
    if flag.find('valid') != -1:
        np.savez('%svis_np_%s_%d.npz'%(config['log'], flag, epoch), h0=h0s_np, h1=h1s_np, pred0s=pred0s_np,
                pred1s=pred1s_np, xs=xs_np, lens=les_np, ys=ys_np, preds=pred_chooses.numpy(), dict=id2token_dict)
    elif flag.find('test') != -1:
        np.savez('%svis_np_%s_%d.npz'%(config['log'],flag, epoch), h0=h0s_np, h1=h1s_np, pred0s=pred0s_np,
                pred1s=pred1s_np, xs=xs_np, lens=les_np, ys=ys_np, preds=pred_chooses.numpy(), dict=id2token_dict)
    elif flag.find('train') != -1:
        np.savez('%svis_np_%s_%d.npz'%(config['log'],flag, epoch), h0=h0s_np, h1=h1s_np, pred0s=pred0s_np,
                pred1s=pred1s_np, xs=xs_np, lens=les_np, ys=ys_np, preds=pred_chooses.numpy(), dict=id2token_dict)

    return total_diff_loss, total_disc_loss, total_enpy, total_right_counts,\
             total_ratios, pred_c_nums, pred_chooses, res_data


def train_epoch(epoch, optims, models, train_data,
                id2token_dict, start_time, flag=None):

    # for log print 
    tokens_data = []
    lens_data = []
    labels_data = []

    # tons of init
    prob_pred_datas = {}
    losses_datas = {}
    attns_datas = {}
    res_data = []
    total_losses = {}
    total_diff_losses, total_disc_losses, total_enpys = 0.0, 0.0, 0.0
    total_count = 0.0
    total_right_counts = {}
    report_losses = {}
    report_diff_losses, report_disc_losses, report_enpys = 0.0, 0.0, 0.0
    report_count = 0.0
    report_right_counts = {}
    report_ratios = {}
    total_ratios = {}
    pred_c_nums = {}

    report_c0_losses = 0.0

    for _i in range(config['hop'] + 1): # +1 for cls
        prob_pred_datas[_i] = []
        losses_datas[_i] = []
        attns_datas[_i] = []
        total_losses[_i], total_right_counts[_i] = 0, 0
        report_losses[_i], report_right_counts[_i] = 0, 0
        report_ratios[_i], total_ratios[_i], pred_c_nums[_i] = 0, 0, 0

    train_indices = train_data.shuffle() # get the indices after shuffle

    # training mode
    for m in models:
        models[m].train()
        models[m].zero_grad()
    
    for i in range(len(train_data)):
        # print freeze
        batch_idx = i
        batch = train_data[batch_idx]
        x_data = batch[0]
        lens = x_data[1]
        y_data = batch[1]
        batch_size = len(y_data)

        tokens_data += [x_data[0].cpu().data]
        lens_data += x_data[1]
        labels_data += [y_data.cpu().data]

        ## C0 stuff
        # memory part
        memory = models['memory'](x_data)

        cs_losses = []
        cs_outputs = []
        cs_attns = []
        cs_hs = []
        for _i in range(config['hop']):
            if _i == 0:
                c_im1_context_t0s = init_context0(batch_size, config['mem_size'])
                c_im1_ch_t0s = init_cht0(config['layer_c'], batch_size, config['rnn_size_c'])
                # connector for first classifier
                c_im1_trigger_t0s = models['connector'](memory, lens)[0] # 0 for h

            c_i_outputs, c_i_attns, c_i_ch_t1s, c_i_context_t1s = models['C%d'%_i](memory,
                                                lens,
                                                c_im1_trigger_t0s,
                                                c_im1_context_t0s,
                                                c_im1_ch_t0s)
            c_i_losses = criterion(c_i_outputs, y_data, False)
            c_i_loss = torch.sum(c_i_losses) / batch_size

            cs_hs.append(c_i_ch_t1s[1][0])
            c_im1_context_t0s = c_i_context_t1s
            c_im1_ch_t0s = (c_i_ch_t1s[0], c_i_ch_t1s[1])
            c_im1_trigger_t0s = c_i_ch_t1s[1][0] # 0 for top layer

            cs_losses.append(c_i_losses)
            cs_outputs.append(c_i_outputs)
            cs_attns.append(c_i_attns)

        ## differentiated loss 
        # Beta distribution
        beta_value = (Beta(config['alpha'], config['beta'])).item()
        c0_ps = cs_outputs[0].gather(1, y_data.view(-1,1))
        reg = (1/beta_value) * torch.pow(c0_ps, config['alpha']-1)*torch.pow(1-c0_ps, config['beta']-1)
        c1_optim_losses = reg * cs_losses[1]
        c0_entropy = torch.sum(-cs_outputs[0]*torch.log(cs_outputs[0]), 1).unsqueeze(1)

        # this is inconsistent with the 'differentiated loss' in the papaer, since pytorch=0.2 has a bug, 
        # torch.mean(a+b)!=torch.mean(a)+torch.mean(b), so we add confidence pernalty directly
        other_loss = torch.mean(c1_optim_losses - config['enpy'] * c0_entropy)
        differentiated_loss = torch.mean(cs_losses[0] + c1_optim_losses)

        ## discriminator
        d_outputs = models['discriminator'](cs_hs)
        loss_dis = 1 - nn.Softmax()(torch.cat([c_losses/config['temp'] for c_losses in cs_losses], 1))
        d_loss = torch.mean(loss_dis*torch.pow(1-d_outputs, 0)*(-torch.log(d_outputs)))

        # this is inconsistent with the 'differentiated loss' in the papaer, since pytorch=0.2 has a bug, 
        # torch.mean(a+b)!=torch.mean(a)+torch.mean(b), so we add confidence pernalty directly
        loss = torch.mean(cs_losses[0]) + other_loss + d_loss
        torch.set_printoptions(precision=10)
        # print loss

        # print x_data[0][0]
        # print memory[0][0][:10]
        # print cs_losses[0]

        # sys.exit()

        loss.backward()
        for key in models:
            optims[key].step()

        ## ton of print
        total_count += batch_size
        report_count += batch_size
        total_diff_losses += differentiated_loss.data[0] * batch_size
        total_disc_losses += d_loss.data[0] * batch_size
        total_enpys += torch.mean(c0_entropy).data[0] * batch_size
        report_diff_losses += differentiated_loss.data[0] * batch_size
        report_disc_losses += d_loss.data[0] * batch_size
        report_enpys += torch.mean(c0_entropy).data[0] * batch_size
        report_c0_losses += torch.mean(cs_losses[0]).data[0] * batch_size
        for _i in range(config['hop']):
            # for print
            attns_datas[_i] += [cs_attns[_i].cpu().data]
            prob_pred_datas[_i] += [cs_outputs[_i].cpu().data]
            losses_datas[_i] += [cs_losses[_i].cpu().data]
            
            right = (y_data.squeeze(1)==cs_outputs[_i].max(1)[1]).sum().data[0]
            loss_float = cs_losses[_i].data[0]

            total_right_counts[_i] += right
            total_losses[_i] += loss_float*batch_size

            pred_c_nums[_i] += (d_outputs.data.max(1)[1] == _i).sum()
            gold_c_num = (loss_dis.data.max(1)[1] == _i).sum()
            report_ratios[_i] += gold_c_num
            total_ratios[_i] += gold_c_num

        # discriminator
        right = (loss_dis.max(1)[1]==d_outputs.max(1)[1]).sum().data[0]
        loss_float = d_loss.data[0]
        total_right_counts[_i + 1] += right
        total_losses[_i + 1] += loss_float*batch_size

        if i % config['interval'] == -1 % config['interval']:
            _s = " diff %6.4f, disc %6.4f, enpy %6.4f, tol %6.4f, C0 %6.4f"%(report_diff_losses/report_count,
                                                       report_disc_losses/report_count,
                                                       -config['enpy']*report_enpys/report_count,
                                                       (report_diff_losses+report_disc_losses-config['enpy']*report_enpys)/report_count,
                                                       report_c0_losses/report_count
                                                       )
            report_diff_losses, report_disc_losses, report_enpys = \
                                                    0.0, 0.0, 0.0
            report_c0_losses = 0.0

            ratio = ''
            for _i in range(1, config['hop']):
                ratio += ' %4.3f'%(report_ratios[_i]/report_count)
                report_ratios[_i] = 0
            _s += (' (ratio%s); ' % ratio)
            report_count = 0.0
            print "Epoch %2d %3d/%3d;%s%4.0fs elapsed" %\
            (epoch, i+1, len(train_data), _s,
                time.time()-start_time)

    # 平均化
    total_diff_loss = total_diff_losses / total_count
    total_disc_loss = total_disc_losses / total_count
    total_enpy = total_enpys / total_count
    for _i in range(config['hop'] + 1):
        total_losses[_i] /= total_count
        total_right_counts[_i] /= total_count
        if _i < config['hop']:
            total_ratios[_i] /= total_count
            pred_c_nums[_i] /= total_count

    # logging
    for _i in range(config['hop']):
        # print len(tokens_data), len(lens_data), len(labels_data), len(attns_data)
        res = print_log(_i, config['log'], epoch, tokens_data, lens_data, labels_data,
                                attns_datas[_i], prob_pred_datas[_i], losses_datas[_i], 
                                id2token_dict, flag, train_indices)
        res_data.append(res)

    return total_diff_loss, total_disc_loss, total_enpy, total_right_counts,\
             total_ratios, pred_c_nums, res_data


def train_models(models, train_data, valid_data, test_data, dicts, optims, writer):

    print '\nBegin training...'

    best_valid_acc, best_valid_test_acc, best_epoch = 0, 0, -1
    best_valid_acc1, best_valid_test_acc1, best_epoch1 = 0, 0, -1

    start_time = time.time()
    for epoch in range(config['start_epoch'], config['epochs'] + 1):

        (train_diff_loss, train_disc_loss, train_enpy, train_total_rights, train_ratios,
                train_pred_nums, train_res) = train_epoch(epoch, optims, models,
                                train_data, dicts['id2token'], start_time,
                                flag=config['flag']+'_train')

        merge_score = merge_res(train_res)

        print 'Train: diff %7.4f, disc %6.4f, enpy %6.4f, tol %6.4f, (pred_ratio %s)' % (
                        train_diff_loss,
                        train_disc_loss,
                        -config['enpy']*train_enpy,
                        train_diff_loss+train_disc_loss-config['enpy']*train_enpy,
                        ' '.join(['%4.3f'%(train_pred_nums[x]) for x in train_pred_nums if x != 0 and x != config['hop']]))

        _s = ''
        for _i in range(config['hop'] + 1):
                _flag = 'D' if _i == config['hop'] else 'C%d'%_i # format output
                _s += ' %s %5.4f,' % (_flag, train_total_rights[_i])

        print '       acc %s (ratio %s), sum %6.4f' % (_s, 
                        ' '.join(['%4.3f'%(train_ratios[x]) for x in train_ratios if x != 0 and x != config['hop']]),
                        merge_score
                        )

        if config['visualize']:
            writer.add_scalars('train_data/loss', {"diff":train_diff_loss,
                                             "disc":train_disc_loss,
                                             "enpy":-config['enpy']*train_enpy,
                                             "tol":train_diff_loss+train_disc_loss-config['enpy']*train_enpy},
                                             epoch)

            writer.add_scalars('train_data/acc', {"C0":train_total_rights[0],
                                            "C1":train_total_rights[1],
                                            "D":train_total_rights[2],
                                             "sum":merge_score},
                                             epoch)

        (valid_diff_loss, valid_disc_loss, valid_enpy, valid_total_rights, valid_ratios,
                valid_pred_nums, valid_pred_chooses, valid_res) = \
                            eval(epoch, models, valid_data,
                            dicts['id2token'], flag=config['flag']+'_valid')

        valid_merge_score, (valid_merge_score5, oracle_score, _) = merge_res(valid_res),\
                                        merge_res5(valid_res, valid_pred_chooses, 0.5)

        print 'Valid: diff %7.4f, disc %6.4f, enpy %6.4f, tol %6.4f, (pred_ratio %s)' % (
                        valid_diff_loss,
                        valid_disc_loss,
                        -config['enpy']*valid_enpy,
                        valid_diff_loss+valid_disc_loss-config['enpy']*valid_enpy,
                        ' '.join(['%4.3f'%(valid_pred_nums[x]) for x in valid_pred_nums if x != 0 and x != config['hop']]))

        _s = ''
        for _i in range(config['hop'] + 1):
                _flag = 'D' if _i == config['hop'] else 'C%d'%_i # format output
                _s += ' %s %5.4f,' % (_flag, valid_total_rights[_i])

        print '       acc %s (ratio %s), sum %5.4f, all %5.4f, orak %5.4f' % (_s, 
                        ' '.join(['%4.3f'%(valid_ratios[x]) for x in valid_ratios if x != 0 and x != config['hop']]),
                        valid_merge_score,
                        valid_merge_score5,
                        oracle_score
                        )

        if config['visualize']:
            writer.add_scalars('valid_data/loss', {"diff":valid_diff_loss,
                                             "disc":valid_disc_loss,
                                             "enpy":-config['enpy']*valid_enpy,
                                             "tol":valid_diff_loss+valid_disc_loss-config['enpy']*valid_enpy},
                                             epoch)

            writer.add_scalars('valid_data/acc', {"C0":valid_total_rights[0],
                                            "C1":valid_total_rights[1],
                                            "D":valid_total_rights[2],
                                            "sum":valid_merge_score,
                                            "tol":valid_merge_score5},
                                             epoch)

        (test_diff_loss, test_disc_loss, test_enpy, test_total_rights, test_ratios,
                test_pred_nums, test_pred_chooses, test_res) = \
                            eval(epoch, models, test_data,
                            dicts['id2token'], flag=config['flag']+'_test')

        test_merge_score, (test_merge_score5, oracle_score, _) = merge_res(test_res),\
                                        merge_res5(test_res, test_pred_chooses, 0.5)

        print ' Test: diff %7.4f, disc %6.4f, enpy %6.4f, tol %6.4f, (pred_ratio %s)' % (
                        test_diff_loss,
                        test_disc_loss,
                        -config['enpy']*test_enpy,
                        test_diff_loss+test_disc_loss-config['enpy']*test_enpy,
                        ' '.join(['%4.3f'%(test_pred_nums[x]) for x in test_pred_nums if x != 0 and x != config['hop']]))

        _s = ''
        for _i in range(config['hop'] + 1):
                _flag = 'D' if _i == config['hop'] else 'C%d'%_i # format output
                _s += ' %s %5.4f,' % (_flag, test_total_rights[_i])

        print '       acc %s (ratio %s), sum %5.4f, all %5.4f, orak %5.4f' % (_s, 
                        ' '.join(['%4.3f'%(test_ratios[x]) for x in test_ratios if x != 0 and x != config['hop']]),
                        test_merge_score,
                        test_merge_score5,
                        oracle_score
                        )

        print ''


        if valid_merge_score >= best_valid_acc:
            best_valid_acc = valid_merge_score
            best_epoch = epoch
            best_valid_test_acc = test_merge_score

        if valid_merge_score5 >= best_valid_acc1:
            best_valid_acc1 = valid_merge_score5
            best_epoch1 = epoch
            best_valid_test_acc1 = test_merge_score5

            # models_dict = {}
            # for m in models:
            #     model_state_dict = models[m].module.state_dict() if len(config['gpu']) > 1 \
            #                                                 else models[m].state_dict()
            #     model_state_dict = {k: v for k, v in model_state_dict.items()}
            #     models_dict[m] = model_state_dict

            # checkpoint = {
            #     'models': models_dict,
            #     'config': config,
            #     'vocab': dicts,
            #     'train_data': train_data,
            #     'valid_data': valid_data,
            #     'test_data': test_data
            # }
            # torch.save(checkpoint,
            #            '%s/%s_models_%5.4f.pt' % (config['save_model'], config['flag'], best_valid_test_acc1))

    if config['visualize']:
        writer.export_scalars_to_json("./runs/all_scalars.json")
        writer.close()

    print 'SUM-MAX  -> epoch%2d'%best_epoch
    print '* Best valid acc.: %5.4f'%best_valid_acc
    print '* Test accuracy with best valid acc.: %5.4f'%best_valid_test_acc
    print 'CLS-PICK -> epoch%2d'%best_epoch1
    print '* Best valid acc.: %5.4f'%best_valid_acc1
    print '* Test accuracy with best valid acc.: %5.4f'%best_valid_test_acc1


def main():

     # 读入数据
    dataset = torch.load(config['data'])

    train_data = Dataset(config, dataset['train'], dataset['vocab'])
    valid_data = Dataset(config, dataset['valid'], dataset['vocab'],
                            volatile=True)
    test_data =  Dataset(config, dataset['test'], dataset['vocab'],
                            volatile=True)

    dicts = dataset['vocab']
    print ' * vocabulary size %d' % dicts['size']
    print ' * number of training sentences %d' % len(dataset['train'][0])
    print ' * maximum batch size. %d' % config['batch_size']

    print('Building model...')
    
    memory = Memory(config, dicts['size'])
    nmemParams = sum([p.nelement() for p in memory.parameters()])

    attn_subnets = {}
    for _i in range(config['hop']):
        attn_subnets['C%d'%_i] = Classifier(config)
    nsubParams = config['hop'] * sum([p.nelement() for p in attn_subnets['C0'].parameters()])

    connector = Connector(config) # get the the initial context information
    nconParams = sum([p.nelement() for p in connector.parameters()])

    # # cls model
    discriminator = Discriminator(config)
    ndisParams = sum([p.nelement() for p in discriminator.parameters()])

    # ugly...
    models = OrderedDict()
    models['discriminator'] = discriminator.cuda() if config['gpu'] != -1 else discriminator
    models['C1'] = attn_subnets['C1'].cuda() if config['gpu'] != -1 else attn_subnets['C1']
    models['C0'] = attn_subnets['C0'].cuda() if config['gpu'] != -1 else attn_subnets['C0']
    models['memory'] = memory.cuda() if config['gpu'] != -1 else memory
    models['connector'] = connector.cuda() if config['gpus'] != -1 else connector

    torch.set_printoptions(precision=10)

    for m in models:
        # parameter initialization
        for p in models[m].parameters():
            p.data.uniform_(-config['param_init'], config['param_init'])

    tolParams = nmemParams + nsubParams + nconParams + ndisParams
    print '* number of total parameters: %d' % tolParams

    # use it after the parameter initialization
    if config['pre_word_vecs'] != 'None':
        memory.load_pre_word_vecs()

    optims = {}
    for m in models:
        optim_i = Optim(
                config['optim'], config['learning_rate'], config['max_grad_norm'],
                lr_decay=config['learning_rate_decay'],
                start_decay_at=config['start_decay_at'],
                weight_decay=config['weight_decay'])
        optim_i.set_parameters(models[m].parameters())
        optims[m] = optim_i

    writer = None
    if config['visualize']:
        writer = SummaryWriter()

    train_models(models, train_data, valid_data, test_data, dicts, optims, writer)


if __name__ == "__main__":
    main()
