from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_cm_from_array(tar, pre, events, show_ylabel=True, normalize=True, title='',
            xlabel='Predicted labels', ylabel='True labels', color_xticklabels=None, cmap="Blues"):

    # calculate confusion matrix
    tar = np.concatenate((tar, np.array(range(len(events)))), axis=0)
    pre = np.concatenate((pre, np.array(range(len(events)))), axis=0)

    cm = confusion_matrix(y_true=tar, y_pred=pre)
    cm -= np.eye(len(events)).astype(np.int32)

    plot_cm(cm, events, show_ylabel=show_ylabel, normalize=normalize, title=title,
            xlabel=xlabel, ylabel=ylabel, color_xticklabels=color_xticklabels, cmap=cmap)


def plot_cm(cm, events, show_ylabel=True, title='', fontsize=6, marginals=False,
            xlabel='Predicted labels', ylabel='True labels', color_xticklabels=None, cmap="Blues", figsize=(2., 2.),
            vmin=None, vmax=None, decimals=2, decimals_marg=3, diag_color=''):

    # figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=len(events) + 1, ncols=len(events) + 1)
    ax = fig.add_subplot(gs[:len(events), :len(events)])

    fmt = f".{decimals}f"

    if len(diag_color) == 0:
        sns.heatmap(cm, annot=True, ax=ax, cbar=False, annot_kws={"size": fontsize}, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax)
    else: 
        # Create a mask for the diagonal
        mask = np.eye(len(events), dtype=bool)

        # Plot the heatmap without the diagonal
        sns.heatmap(cm, annot=True, ax=ax, cbar=False, annot_kws={"size": fontsize}, fmt=fmt, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask)

        # Plot the diagonal elements with the specified color
        for i in range(len(events)):
            ax.text(i + 0.5, i + 0.5, f"{cm[i, i]:{fmt}}", ha='center', va='center', color=diag_color, fontsize=fontsize)

    if marginals:
        fmt_margs = f".{decimals_marg}f"

        # metrics
        cm_precision = np.reshape(cm[np.eye(len(events), dtype=bool)] / (1e-9 + cm.sum(axis=0)), (1, len(events)))
        cm_recall = np.reshape(cm[np.eye(len(events), dtype=bool)] / (1e-9 + cm.sum(axis=1)), (len(events), 1))
        cm_acc = np.reshape(cm.trace() / (1e-9 + cm.sum()), (1, 1))
        # cm_f1 = np.reshape(np.mean([2 * (pr * re) / (1e-9 + pr + re) for pr, re in zip(cm_precision, cm_recall)]), (1, 1))
        # cm_kappa_w = np.reshape(np.round(cohen_kappa_score(tar, pre, weights='linear') * 100) / 100, (1, 1))

        # axis
        ax_re = fig.add_subplot(gs[:len(events), -1])
        ax_pr = fig.add_subplot(gs[-1, :len(events)])
        ax_acc = fig.add_subplot(gs[-1, -1])

        sns.heatmap(cm_recall, annot=True, ax=ax_re, cbar=False, annot_kws={"size": fontsize}, fmt=fmt_margs, vmin=0, vmax=1, cmap=cmap)
        sns.heatmap(cm_precision, annot=True, ax=ax_pr, cbar=False, annot_kws={"size": fontsize}, fmt=fmt_margs, vmin=0, vmax=1, cmap=cmap)
        sns.heatmap(cm_acc, annot=True, ax=ax_acc, cbar=False, annot_kws={"size": fontsize}, fmt=fmt_margs, vmin=0, vmax=1, cmap=cmap)

        ax_pr.xaxis.set_tick_params(which='major', size=0, width=0, direction='in')
        ax_pr.yaxis.set_tick_params(which='major', size=0, width=0, direction='in')
        ax_pr.xaxis.set_ticklabels([], fontsize=fontsize)
        ax_pr.yaxis.set_ticklabels(['Pr'], fontsize=fontsize)

        ax_re.xaxis.tick_top()
        ax_re.xaxis.set_label_position('top')
        ax_re.yaxis.set_tick_params(which='major', size=0, width=0, direction='in')
        ax_re.xaxis.set_tick_params(which='major', size=0, width=0, direction='in')
        ax_re.yaxis.set_ticklabels([], fontsize=fontsize)
        ax_re.xaxis.set_ticklabels(['Re'], fontsize=fontsize)
        # ax_pr.xaxis.set_ticklabels(events, fontsize=fontsize)

        ax_acc.yaxis.set_tick_params(which='major', size=0, width=0, direction='in')
        ax_acc.xaxis.set_tick_params(which='major', size=0, width=0, direction='in')
        ax_acc.yaxis.set_ticklabels([], fontsize=fontsize)
        ax_acc.xaxis.set_ticklabels(['Acc'], fontsize=fontsize)

    # title
    ax.set_title(title, fontsize=fontsize + 3)
    
    # xaxis
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.xaxis.set_tick_params(which='major', size=0, width=1, direction='in')
    ax.xaxis.set_ticklabels(events, fontsize=fontsize)

    # yaxis
    ax.yaxis.set_tick_params(which='major', size=0, width=1, direction='in')
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.yaxis.set_ticklabels(events, fontsize=fontsize)
    else:
        plt.ylabel('')
        ax.yaxis.set_ticklabels([], fontsize=fontsize)
    
    # return fig