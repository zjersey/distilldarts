import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_search_csv(filename):
    # read csv file
    a = np.array(pd.read_csv(filename,header=1))
    epoch = a[:,0]
    val_loss = a[:,1]
    val_acc = a[:,2]
    argmax_edge_acc = a[:,3]
    argmax_edge_loss = a[:,4]
    sample_edge_acc = a[:,5]
    sample_edge_loss = a[:,6]
    argmax_operator_acc = a[:,7]
    argmax_operator_loss = a[:,8]
    sample_operator_acc = a[:,9]
    sample_operator_loss = a[:,10]

    save_dir = os.path.join(*filename.split('/')[:-1])
    # plot epoch-val_loss
    line = plt.plot(epoch, val_loss)
    plt.title('epoch-val_loss')
    save_path = os.path.join(save_dir, 'epoch-val_loss.png')
    plt.savefig(save_path)
    plt.show()
    # plot epoch-val_acc
    line = plt.plot(epoch, val_acc)
    plt.title('epoch-val_acc')
    save_path = os.path.join(save_dir, 'epoch-val_acc.png')
    plt.savefig(save_path)
    plt.show()
    # plot epoch-argmax-sample
    line = plt.plot(epoch, argmax_edge_loss, 'bo-', epoch, sample_edge_loss, 'rx-')
    plt.legend(('argmax_edge_loss', 'sample_edge_loss'), loc='upper right')
    plt.title('epoch-argmax-sample_loss')
    save_path = os.path.join(save_dir, 'epoch-argmax-sample_loss.png')
    plt.savefig(save_path)
    plt.show()
    # plot epoch-argmax-sample
    line = plt.plot(epoch, argmax_edge_acc, 'bo-', epoch, sample_edge_acc, 'rx-')
    plt.legend(('argmax_edge_acc', 'sample_edge_acc'), loc='upper right')
    plt.title('epoch-argmax-sample_acc')
    save_path = os.path.join(save_dir, 'epoch-argmax-sample_acc.png')
    plt.savefig(save_path)
    plt.show()

    


if __name__ == '__main__':
  plot_search_csv('search-EXP-20190917-114211/results_of_7q/results.csv')

