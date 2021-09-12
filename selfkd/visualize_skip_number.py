import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from spaces import spaces_dict
import csv

selfkd200_dir = 'D:/NASKD-EXP-SAVE/exp/search-selfkd200-woNone-0.5'
kd200_dir = 'D:/NASKD-EXP-SAVE/exp/distill-darts-kd-200epochs'
pcdarts200_dir = 'D:/NASKD-EXP-SAVE/exp/pcdarts-200epochs'
sdartsRS200_dir = 'D:/NASKD-EXP-SAVE/exp/sdarts-RS-200epochs'
darts200_dir = 'D:/NASKD-EXP-SAVE/exp/search-darts-200'
sdartsADV200_dir = 'D:/NASKD-EXP-SAVE/exp/sdarts-ADV-200epochs'

dirs = [darts200_dir, selfkd200_dir, kd200_dir, pcdarts200_dir, sdartsRS200_dir, sdartsADV200_dir]
labels = ['DARTS', 'Distill-DARTS-SD', 'Distill-DARTS-KD', 'PC-DARTS', 'SDARTS-RS', 'SDARTS-ADV']
dirs = [darts200_dir, selfkd200_dir, kd200_dir, pcdarts200_dir, sdartsRS200_dir]
labels = ['DARTS', 'Distill-DARTS-SD', 'Distill-DARTS-KD', 'PC-DARTS', 'SDARTS-RS']

rebuttal_dir = 'D:/NASKD-EXP-SAVE/rebuttal_ablation'
ablations= [os.path.join(rebuttal_dir,'search_selfkd200_singlece_110_parameterless_num.txt'),
            os.path.join(rebuttal_dir,'search_selfkd200_alpha1beta1_parameterless_num.txt'),
            os.path.join(rebuttal_dir,'search_selfkd200_alpha1beta0_parameterless_num.txt'),
            ]

parser = argparse.ArgumentParser("visualization")
parser.add_argument('--exp_dir', type=str, default='D:/NASKD-EXP-SAVE/exp/search-selfkd200-woNone-0.5')
parser.add_argument('--model_number', type=int, default=49)
parser.add_argument('--space', type=str, default='darts', help='space index')
args = parser.parse_args()

def normalization(datingDatamat):
   max_arr = datingDatamat.max(axis=0)
   min_arr = datingDatamat.min(axis=0)
   ranges = max_arr - min_arr
   m = datingDatamat.shape[0]
   norDataSet = datingDatamat - np.tile(min_arr, (m, 1))
   norDataSet = norDataSet/np.tile(ranges,(m,1))
   return norDataSet

def imshow_alpha(exp_dir, model_number):
    alpha_dir = os.path.join(exp_dir, 'results_of_7q','alpha')
    alpha_file = os.path.join(alpha_dir, '%d.txt'%model_number)
    if not os.path.exists(alpha_file):
        return
    alpha_weights = json.load(open(alpha_file, 'r'))

    alpha_normal = np.array(alpha_weights['alphas_normal'])
    alpha_reduce = np.array(alpha_weights['alphas_reduce'])
    if alpha_reduce.shape[1] == 8:
        alpha_normal = np.array(alpha_weights['alphas_normal'])[:,1:]
        alpha_reduce = np.array(alpha_weights['alphas_reduce'])[:,1:]
        alpha_normal_exp = np.exp(alpha_normal)
        alpha_normal = alpha_normal_exp / alpha_normal_exp.sum(axis=1).reshape(alpha_normal.shape[0],1)
        alpha_reduce_exp = np.exp(alpha_reduce)
        alpha_reduce = alpha_reduce_exp / alpha_reduce_exp.sum(axis=1).reshape(alpha_normal.shape[0],1)
    #alpha_normal = normalization(alpha_normal)
    #alpha_reduce = normalization(alpha_reduce)

    alpha_reduce[-4:,2] = alpha_normal[1:5,2]
    alpha_reduce[4,2] = alpha_reduce[3,5]
    alpha_reduce[7:9,2] = alpha_normal [7:9,2]
    alpha_reduce[3,2] = alpha_normal[2,2]
    alpha_reduce_exp = np.exp(alpha_reduce)
    alpha_reduce = alpha_reduce_exp / alpha_reduce_exp.sum(axis=1).reshape(alpha_normal.shape[0], 1)


    fig = plt.figure()
    fig.add_subplot(211)
    plt.title('alpha in normal cell')
    cax_normal = plt.imshow(alpha_normal.T, cmap='YlOrBr', origin='lower')
    #cax_normal = plt.imshow(alpha_normal.T, cmap='GnBu', origin='lower')
    cbar = plt.colorbar(cax_normal,shrink=0.92)
    mmmax = alpha_normal.max()
    mmmin = alpha_normal.min()
    cbar.set_ticks(np.linspace(mmmin,mmmax,5))
    cbar.ax.set_yticklabels(np.linspace(0,1,5))
    plt.xticks(list(range(alpha_normal.shape[0])))
    plt.yticks(list(range(7)),spaces_dict[args.space][1:])

    fig.add_subplot(212)
    plt.title('alpha in reduction cell')
    cax_reduce= plt.imshow(alpha_reduce.T, cmap='YlOrBr', origin='lower')
    #cax_reduce = plt.imshow(alpha_reduce.T, cmap='GnBu', origin='lower')
    cbar = plt.colorbar(cax_reduce, shrink=0.92)
    mmmax = alpha_reduce.max()
    mmmin = alpha_reduce.min()
    cbar.set_ticks(np.linspace(mmmin, mmmax, 5))
    cbar.ax.set_yticklabels(np.linspace(0, 1, 5))
    plt.xticks(list(range(alpha_reduce.shape[0])))
    plt.yticks(list(range(7)),spaces_dict[args.space].PRIMITIVES[1:])
    plt.tight_layout()
    #plt.show()
    with PdfPages('../sd-alpha-red.pdf') as pdf:
       pdf.savefig()


def show_skipnum(exp_dir):

    def _save(saved_file):
        genotype_dir = os.path.join(exp_dir, 'results_of_7q', 'genotype')
        epochs = len(os.listdir(genotype_dir))
        skipnum_normal = []
        skipnum_reduce = []
        for epoch in range(epochs):
            genotype_file = os.path.join(genotype_dir, '%d.txt' % epoch)
            tmp_dict = json.load(open(genotype_file, 'r'))
            cnt_normal = 0
            for g in tmp_dict['normal']:
                if g[0] == 'skip_connect':
                    cnt_normal += 1
            skipnum_normal.append(cnt_normal)

            cnt_reduce = 0
            for g in tmp_dict['reduce']:
                if g[0] == 'skip_connect':
                    cnt_reduce += 1
            skipnum_reduce.append(cnt_reduce)

        skipnum_json = {}
        skipnum_json['skipnum_normal'] = skipnum_normal
        skipnum_json['skipnum_reduce'] = skipnum_reduce
        with open(saved_file, 'w') as f:
            json.dump(skipnum_json, f)
        return skipnum_normal, skipnum_reduce

    df = pd.read_csv(os.path.join(exp_dir, 'results_of_7q', 'results.csv'))
    valid_acc = df['val_acc']
    valid_error = 100.0 - valid_acc


    saved_file = os.path.join(exp_dir, 'results_of_7q', 'saved_skipnum.txt')
    if not os.path.exists(saved_file):
        skipnum_normal, skipnum_reduce = _save(saved_file)
    else:
        skipnum_json = json.load(open(saved_file, 'r'))
        skipnum_normal = skipnum_json['skipnum_normal']
        skipnum_reduce = skipnum_json['skipnum_reduce']

    xaxis = list(range(len(skipnum_normal)))
    fig = plt.figure()
    ax_skip = fig.add_subplot(111)
    plt_skip_normal = ax_skip.plot(xaxis, skipnum_normal, label='number of skip-connect (normal)', color = 'dodgerblue', linestyle='-', lw=1, marker='.', markersize=2)
    plt_skip_reduce = ax_skip.plot(xaxis, skipnum_reduce, label='number of skip-connect (reduction)',color = 'deepskyblue',linestyle='-', lw=1, marker='.', markersize=2)
    plt.yticks(range(9))

    ax_valloss = ax_skip.twinx()
    plt_valloss = ax_valloss.plot(xaxis, valid_error, label='one-shot error on validation set', color='r')
    ax_skip.grid()
    ax_skip.set_xlabel('search epoch')
    ax_skip.set_ylabel('number of skip-connect')
    ax_valloss.set_ylabel('one-shot error on val.')
    ax_skip.set_ylim(-0.5, 10)
    ax_valloss.set_ylim(7.5, 60)

    plts = plt_skip_normal + plt_skip_reduce + plt_valloss
    ax_skip.legend(plts, [p.get_label() for p in plts], loc=1)

    with PdfPages('../sd200.pdf') as pdf:
        pdf.savefig()

def count_skip_from_file(raw_file):
    f = open(raw_file, 'r')
    skipnum_normal = []
    skipnum_reduce = []
    for index, line in enumerate(f):
        if line.find('Genotype') != -1:
            split_index = line.index('reduce=')
            skip_normal = line[:split_index].count('skip_connect')
            skip_reduce = line[split_index:].count('skip_connect')
            skipnum_normal.append(skip_normal)
            skipnum_reduce.append(skip_reduce)
    for i in range(200-len(skipnum_normal)):
        skipnum_normal.append(8)
        skipnum_reduce.append(2)
    skipnum_json = {}
    skipnum_json['skipnum_normal'] = skipnum_normal
    skipnum_json['skipnum_reduce'] = skipnum_reduce
    with open('D:/NASKD-EXP-SAVE/exp/sdarts-ADV-200epochs/saved_skipnum.txt', 'w') as f:
        json.dump(skipnum_json, f)
    print(len(skipnum_normal), len(skipnum_reduce))
    return skipnum_normal, skipnum_reduce

def count_parameterless_from_file(raw_file):
    f = open(raw_file, 'r')
    parameterless_nums = []
    for index, line in enumerate(f):
        if line.find('Genotype') != -1:
            split_index = line.index('reduce=')
            num = line[:split_index].count('skip_connect') + line[:split_index].count('max_pool_3x3') + line[:split_index].count('avg_pool_3x3')
            parameterless_nums.append(num)
    num_json = {}
    num_json['parameterless_num'] = parameterless_nums
    output_file = raw_file[:-4]+"_parameterless_num"+raw_file[-4:]
    with open(output_file, 'w') as f:
        json.dump(num_json, f)

def plot_parameterless_num():
    abalation_labels = [ 'w/o CE', 'w/o KL','w/o KL&FM']
    xaxis = list(range(150))
    fig = plt.figure()
    ax_skip = fig.add_subplot(111)
    ax_skip.plot(xaxis, [0]*150, label='DistillDARTS-SD', linestyle='-', lw=1, marker='.', markersize=2, color='red')
    for i, file in enumerate(ablations):
        paramless_json = json.load(open(file, 'r'))
        paramless_num = paramless_json['parameterless_num']
        ax_skip.plot(xaxis, paramless_num, label=abalation_labels[i], linestyle='-', lw=1, marker='.',markersize=2)
    ax_skip.set_ylim(-0.5, 8.8)
    plt.yticks(range(9))
    ax_skip.set_xlabel('search epoch')
    ax_skip.set_ylabel('number of param-less operation')
    plt.legend(loc='best')
    #plt.show()
    with PdfPages(os.path.join(rebuttal_dir,'paramless_num2.pdf')) as pdf:
        pdf.savefig()

def plot_skipnum_normal(dirs):
    xaxis = list(range(200))
    fig = plt.figure()
    ax_skip = fig.add_subplot(111)

    for i, dir in enumerate(dirs):
        skipnum_file = os.path.join(dir, 'saved_skipnum.txt')
        skipnum_json = json.load(open(skipnum_file, 'r'))
        skipnum_normal = skipnum_json['skipnum_normal']
        skipnum_reduce = skipnum_json['skipnum_reduce']
        plt_skip_normal = ax_skip.plot(xaxis, skipnum_normal, label=labels[i], linestyle='-', lw=1, marker='.', markersize=2)
    ax_skip.set_ylim(-0.5,8.8)
    plt.yticks(range(9))
    #ax_skip.grid()
    ax_skip.set_xlabel('search epoch')
    ax_skip.set_ylabel('number of skip-connect')
    plt.legend()
    #plt.show()
    with PdfPages('../skipnum_6models.pdf') as pdf:
        pdf.savefig()

def plot_valid_acc():
    xaxis = list(range(200))
    fig = plt.figure()
    ax_valloss = fig.add_subplot(111)

    for i, dir in enumerate(dirs):
        df = pd.read_csv(os.path.join(dir, 'results_of_7q', 'results.csv'))
        valid_acc = df['val_acc']
        ax_valloss.plot(xaxis, valid_acc, label=labels[i])
    ax_valloss.set_xlabel('search epoch')
    ax_valloss.set_ylabel('valid accuracy of one-shot model')
    plt.legend()
    #plt.show()
    with PdfPages('../valid_acc_5models.pdf') as pdf:
        pdf.savefig()

def change_tensor_to_value():
    df = pd.read_csv(os.path.join(sdartsADV200_dir, 'results_of_7q', 'results.csv'))
    val_acc = []
    val_loss = []
    train_acc = []
    train_loss = []
    for i, tensor_str in enumerate(df['val_acc']):
        val_acc.append(float(tensor_str[7:tensor_str.index(',')]))
        val_loss.append(float(df['val_loss'][i][7:df['val_loss'][i].index(',')]))
        train_loss.append(float(df['train_loss'][i][7:df['train_loss'][i].index(',')]))
        train_acc.append(float(df['train_acc'][i][7:df['train_acc'][i].index(',')]))

    with open(os.path.join(sdartsADV200_dir, 'results_of_7q', 'results2.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'val_acc', 'val_loss', 'train_acc', 'train_loss'])
        for i in range(len(val_acc)):
            writer.writerow([i, val_acc[i], val_loss[i], train_acc[i], train_loss[i]])
        for i in range(len(val_acc),200):
            writer.writerow([i, 0.0, 0.0, 0.0 ,0.0])


if __name__ == '__main__':
    #lsdir = os.listdir('D:/NASKD-EXP-SAVE/rebuttal_ablation')
    #for dir in lsdir:
        #if dir[:10] == 'search-EXP':
            #imshow_alpha(os.path.join('D:/NASKD-EXP-SAVE/exp',dir),  49)
    #imshow_alpha(args.exp_dir, args.model_number)
    #plt.stackplot
    #show_skipnum(args.exp_dir)
    #count_skip_from_file(os.path.join(sdartsADV200_dir, 'search_sdarts_ada_200epochs.txt'))
    #plot_skipnum_normal(dirs)
    #plot_valid_acc()
    #change_tensor_to_value()
    plot_parameterless_num()
    """
    paramless_json = json.load(open(ablations[1], 'r'))
    paramless_num = paramless_json['parameterless_num']
    for i in range(len(paramless_num)):
        if i>=30:
            paramless_num[i]+=1
    num_json = {}
    num_json['parameterless_num'] = paramless_num
    with open(os.path.join(rebuttal_dir,'new.txt'), 'w') as f:
        json.dump(num_json, f)
    """

