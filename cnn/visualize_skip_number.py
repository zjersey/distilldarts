import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import genotypes
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


parser = argparse.ArgumentParser("visualization")
parser.add_argument('--exp_dir', type=str, default='D:/NASKD-EXP-SAVE/exp/search-selfkd200-woNone-0.5')
parser.add_argument('--model_number', type=int, default=49)
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
    plt.yticks(list(range(7)),genotypes.PRIMITIVES[1:])

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
    plt.yticks(list(range(7)), genotypes.PRIMITIVES[1:])
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




if __name__ == '__main__':
    #lsdir = os.listdir('D:/NASKD-EXP-SAVE/exp')
    #for dir in lsdir:
        #if dir[:10] == 'search-EXP':
            #imshow_alpha(os.path.join('D:/NASKD-EXP-SAVE/exp',dir),  49)
    #imshow_alpha(args.exp_dir, args.model_number)
    #plt.stackplot
    show_skipnum(args.exp_dir)