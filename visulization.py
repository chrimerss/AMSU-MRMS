import numpy as np
import matplotlib as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def confusion_mtx_colormap(mtx, xnames, ynames, charlabel= "", **figkwargs):
    '''
    Generate a figure that plots a colormap of a matrix
    Args:
    -------------------
    :mtx - matrix of values
    :xnames - list of x tick names
    :ynale - list of y tick 
    :figkwargs - dict plt.imshow key arguments

    Output:
    -------------------
    fig, ax
    '''

    nxvars = mtx.shape[1]
    nyvars = mtx.shape[0]

    fig, ax= plt.subplots()
    im = ax.imshow(mtx, cmap='summer', **figkwargs)
    if not charlabel == "":
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(charlabel, roration=-90, va='bottom')

    ax.set_xticks(range(nxvars))
    ax.set_yticks(range(nyvars))
    ax.set_xticklabels(xnames)
    ax.set_yticklabels(ynames)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("Actual Labels")

    #Rotate the tick labels and set thier alignment
    plt.setp(ax.get_xticklabels(), rotation=45,
            ha="right", rotation_mode="anchor")
    
    #Loop over data dimensions and create iext annotations.
    lbl = np.array([['TN', 'FP'], ['FN', 'TP']])
    for i in range(nyvars):
        for j in range(nxvars):
            text= ax.text(j, i, "%a = %.3f"%(lbl[i,j], mtx[i,j]),
                        ha="center", va="center", color="k")

    return fig, ax

def ks_roc_plot(targets, scores, **figkwargs):
    '''
    Generate a figure that plots a colormap of a matrix

    Args:
    ------------------
    :targets - 
    :scores - 
    :figkwargs - dict plt.plot key arguments
    '''

    fpr, tpr, thresholds = roc_curve(targets, scores)
    auc_res = auc(fpr, tpr)

    #Generate KS plot
    fig, ax= plt.subplots(1,2)
    axs= ax.ravel()
    ax[0].plot(thresholds, tpr, color='b')
    ax[0].plot(thresholds, fpr, color='r')
    ax[0].plot(thresholds, tpr-fpr, color='g')
    ax[0].invert_xaxis()
    ax[0].set_xlabel('threshold', fontsize=15)
    ax[0].set_ylabel('fraction', fontsize=15)
    ax[0].legend(['TPR', 'FPR', 'K-S distance'], fontsize=15)

    #Generate ROC curve plot
    ax[1].plot(fpr, tpr, color='b')
    ax[1].plot([0,1], [0,1], 'r--')
    ax[1].set_xlabel('FPR', fontsize=15)
    ax[1].set_ylabel('TPR', fontsize=15)
    ax[1].set_aspect('equal', 'box')
    auc_text= ax[1].text(.05, .95, "AUC = %.4f"%auc_res, color='k', fontsize=15)

    return fpr, tpr, thresholds, auc, fig, axes
