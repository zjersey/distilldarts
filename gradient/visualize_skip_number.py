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

#dirs = [darts200_dir, selfkd200_dir, kd200_dir, pcdarts200_dir, sdartsRS200_dir, sdartsADV200_dir]
#labels = ['DARTS', 'Distill-DARTS-SD', 'Distill-DARTS-KD', 'PC-DARTS', 'SDARTS-RS', 'SDARTS-ADV']
dirs = [darts200_dir, selfkd200_dir, kd200_dir, sdartsRS200_dir, sdartsADV200_dir]
labels = ['DARTS', 'Distill-DARTS-SD', 'Distill-DARTS-KD', 'SDARTS-RS', 'SDARTS-ADV']
colors = ['#50514F', '#F8766D', '#FA7921', '#5bc0eb', '#619CFF']


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
    plt.show()
    #with PdfPages('../sd-alpha-red.pdf') as pdf:
       #pdf.savefig()



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

def plot_skipnum_normal(dirs):
    xaxis = list(range(200))
    fig = plt.figure()
    ax_skip = fig.add_subplot(111)
    #colors = ['#00BA38', '#F8766D', '#F564E3',  '#00BFC4', '#619CFF']
    #colors = ['#E41A1C', '#377E88', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']
    colors = ['#50514F', '#F8766D', '#FA7921', '#5bc0eb', '#619CFF']
    for i, dir in enumerate(dirs):
        skipnum_file = os.path.join(dir, 'saved_skipnum.txt')
        skipnum_json = json.load(open(skipnum_file, 'r'))
        skipnum_normal = skipnum_json['skipnum_normal']
        ax_skip.plot(xaxis, skipnum_normal, label=labels[i], linestyle='-', lw=1, marker='.', markersize=2, color = colors[i])
    ax_skip.set_ylim(-0.5,8.8)
    plt.yticks(range(9))
    #ax_skip.grid()
    ax_skip.set_xlabel('search epoch', fontsize=12)
    ax_skip.set_ylabel('number of skip-connect',fontsize=12)
    plt.legend()
    #plt.show()
    with PdfPages('../figure_skipnum.pdf') as pdf:
        pdf.savefig()

def plot_valid_acc():
    xaxis = list(range(200))
    fig = plt.figure()
    ax_valloss = fig.add_subplot(111)

    for i, dir in enumerate(dirs):
        df = pd.read_csv(os.path.join(dir, 'results_of_7q', 'results.csv'))
        valid_acc = df['val_acc']
        ax_valloss.plot(xaxis, valid_acc, label=labels[i], color=colors[i])
    ax_valloss.set_xlabel('search epoch', fontsize=12)
    ax_valloss.set_ylabel('valid accuracy of one-shot model', fontsize=12)
    plt.legend()
    #plt.show()
    with PdfPages('../figure_validacc.pdf') as pdf:
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

def plot_evaluation_acc():
    fig = plt.figure()
    ax_evaluation = fig.add_subplot(211)
    ax_down = fig.add_subplot(212)
    df_acc = pd.read_csv('./evaluation_acc.csv')
    df_size = pd.read_csv('./model_size.csv')
    labels = ['DARTS', 'Distill-DARTS-SD', 'Distill-DARTS-KD', 'SDARTS-RS','SDARTS-ADV']
    for i , label in enumerate(labels):
        values = df_acc[label]
        sizes = df_size[label]
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for j in range(len(values)):
            if values[j]!=0:
                x1.append(25 * j)
                y1.append(values[j])
            x2.append(25*j)
            y2.append(sizes[j])
        ax_evaluation.plot(x1, y1, label=label, marker='*', color=colors[i])
        ax_down.plot(x2, y2, label=label, marker='*',color=colors[i])
    ax_evaluation.set_ylabel('re-training accurac (%)', fontsize=12)
    ax_down.set_xlabel('search epoch', fontsize=12)
    ax_down.set_ylabel('model size (M)', fontsize=12)
    ax_evaluation.legend(loc=3)
    #plt.show()
    with PdfPages('../figure_evaluation_acc.pdf') as pdf:
        pdf.savefig()

def plot_fairdarts_num():
    xaxis = list(range(200))
    fig = plt.figure()
    ax_skip = fig.add_subplot(111)
    # colors = ['#00BA38', '#F8766D', '#F564E3',  '#00BFC4', '#619CFF']
    # colors = ['#E41A1C', '#377E88', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33']
    colors = ['#F8766D', '#FA7921', '#5bc0eb', '#619CFF']
    jsonfile_none = 'fairdarts_none.txt'
    jsonfile_wonone = './fairdarts_WOnone.txt'

    json_wonone = json.load(open(jsonfile_wonone, 'r'))
    num_normal = json_wonone['normal']
    num_reduce = json_wonone['reduce']
    ax_skip.plot(xaxis, num_normal, label='Normal cell', linestyle='-', lw=1, marker='.', markersize=2,
                 color=colors[0])
    ax_skip.plot(xaxis, num_reduce, label='Reduction cell', linestyle='-', lw=1, marker='.', markersize=2,
                 color=colors[3])

    #ax_skip.set_ylim(-0.5, 8.8)
    plt.yticks(range(9))
    ax_skip.grid()
    ax_skip.set_xlabel('search epoch', fontsize=12)
    ax_skip.set_ylabel('number of operations found', fontsize=12)
    plt.legend()
    #plt.show()
    with PdfPages('../fairdarts_num.pdf') as pdf:
        pdf.savefig()

if __name__ == '__main__':
    #lsdir = os.listdir('D:/NASKD-EXP-SAVE/exp')
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
    #plot_evaluation_acc()
    plot_fairdarts_num()