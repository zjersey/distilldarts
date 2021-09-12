import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_average_darts50_selfkd50():
    xaxis = list(range(50))
    fig = plt.figure()
    ax_gradient = fig.add_subplot(111)
    for i in range(8):
        df = pd.read_csv(os.path.join('darts-average', 'run-tensorboard-darts_average_layer%d-average-tag-avg_gradient-darts.csv'%i))
        #df = pd.read_csv(os.path.join('selfkd-average', 'run-average_layer%d-average-tag-avg_gradient-selfkd.csv'%i))
        average_gradients = df['Value']
        ax_gradient.plot(xaxis, average_gradients, label='layer-%d'%i)
    ax_gradient.set_ylim(0, 0.03)
    ax_gradient.legend()
    ax_gradient.set_xlim(-1, 50)
    ax_gradient.set_xlabel('search epoch', fontsize=12)
    ax_gradient.set_ylabel('gradient value (2-norm)', fontsize=12)
    #plt.show()
    with PdfPages('./darts50.pdf') as pdf:
        pdf.savefig()

def plot_average():

    fig = plt.figure()
    ax_gradient = fig.add_subplot(111)
    xaxis = list(range(200))

    for i in range(8):
        #df = pd.read_csv(os.path.join('selfkd15-average', 'run-average_layer%d-average-tag-avg_gradient-selfkd.csv'%i))
        #df = pd.read_csv(os.path.join('selfkd-average', 'run-average_layer%d-average-tag-avg_gradient-selfkd.csv'%i))
        #df = pd.read_csv(os.path.join('kd50-average', 'run-average_layer%d-average-kd-tag-avg_gradient-kd.csv' % i))
        #df = pd.read_csv(os.path.join('darts200-average', 'run-average_layer%d-average-tag-avg_gradient-darts200.csv' % i))
        df = pd.read_csv(os.path.join('selfkd200-average', 'run-average_layer%d-average-tag-avg_gradient-selfkd200.csv' % i))
        #df = pd.read_csv(os.path.join('kd200-average', 'run-average_layer%d-average-kd-tag-avg_gradient-kd.csv' % i))
        #df = pd.read_csv(os.path.join('darts-average', 'run-tensorboard-darts_average_layer%d-average-tag-avg_gradient-darts.csv' % i))
        average_gradients = df['Value']

        if i==0 or i==1:
            average_gradients /= 4
        elif i==2 or i==3 or i==4:
            average_gradients /= 2
        ax_gradient.plot(xaxis, average_gradients, label='layer-%d' % i)
    plt.yticks(np.linspace(0, 0.006 ,7), ['{:.0%}'.format(oi/0.14) for oi in np.linspace(0, 0.14 ,7)])
    ax_gradient.set_xlim(-1,200)
    #ax_gradient.set_ylim(0, 0.006)
    ax_gradient.legend()
    ax_gradient.set_xlabel('search epoch', fontsize=12)
    ax_gradient.set_ylabel('average amplitude value of gradients', fontsize=12)
    #plt.show()
    with PdfPages('./selfkd200.pdf') as pdf:
        pdf.savefig()


if __name__ == '__main__':
    plot_average()
    #plot_average_darts50_selfkd50()