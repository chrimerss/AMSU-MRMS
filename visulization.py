import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as peffects
from matplotlib import cm

from sklearn.metrics import explained_variance_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('/Users/hydrosou/Documents/PlotGallary')
from matplotlibconfig import basic
basic()


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

    fig, ax= plt.subplots(figsize=(8,8))
    im = ax.imshow(mtx, cmap='summer', extent= [-0.5,1.5,-0.5,1.5] ,origin='lower')
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
    lbl = np.array([['TP', 'TN'], ['FP', 'FN']])
    for i in range(nyvars):
        for j in range(nxvars):
            text= ax.text(j, i, "%a = %d"%(lbl[i,j], mtx[i,j]),
                        ha="center", va="center", color="k")

    return fig, ax

def plot_classificationReport(y_pred, y_true, **kwargs):
    '''
    Generate classification report plot including: precision, recall, f1-score, support

    Args:
    -----------------
    :y_pred - numpy array; predicted labels
    :y_true - numpy array; reference
    :kwargs - dict of plot configurations

    Returns:
    -----------------
    :fig - matplotlib.Figure()
    :ax - current axes
    '''
    target_names= ['no rain', 'rain']
    cmap= kwargs.pop('cmap', 'summer')

    if not (isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray)):
        raise ValueError('Expected numpy array input, but get: %s'%str(type(y_pred)))
    _dict= classification_report(y_true, y_pred, output_dict=True)
    print(_dict)
    metrics= [_dict[_class][metric]  for _class in ['0.0', '1.0'] for metric in _dict[_class].keys()]
    metrics= np.array(metrics).reshape(2,4)
    X, Y= (
        np.arange(len(target_names)+1),
        np.arange(5)
        )
    fig, ax= plt.subplots(figsize=(12,6))
    ax.set_ylim(bottom=0, top=metrics.shape[0])
    ax.set_xlim(left=0, right=metrics.shape[1])
    for x in X[:-1]:
        for y in Y[:-1]:
            value= metrics[x,y]
            svalue= "{:0.3f}".format(value)
            
#             base_color= cmap(value)
            cx, cy= 0.5+x, 0.5+y
            ax.text(cy, cx, svalue, va="center", ha="center", color='k')
    im= ax.pcolormesh(
        Y, X, metrics, vmin=0, vmax=1,cmap=cmap, **kwargs, edgecolor='w'
    )
    plt.colorbar(im, ax= ax)
    ax.set_xticks(Y[:-1]+0.5)
    ax.set_xticklabels(list(_dict['0.0'].keys()))
    ax.set_yticks(X[:-1]+0.5)
    ax.set_yticklabels(['no rain', 'rain'], fontsize=15)

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
    axes= ax.ravel()
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



def skillScore(y_true, y_pred, skill='pss'):
    """
    Compute various skill scores
    PARAMS:
        y_true: the true classification label
        y_pred: the classification predicted by the model (must be binary)
        skill: a string used to select a particular skill score to compute
                'pss' | 'hss' | 'bss'
    """
    cmtx = confusion_matrix(y_true, y_pred)
    tn = cmtx[0,0]
    fp = cmtx[0,1]
    fn = cmtx[1,0]
    tp = cmtx[1,1]

    if skill == 'acc': #accuracy
        return float(tp + tn) / (tp + fn + tn + fp)
    if skill == 'pss':
        tpr = float(tp) / (tp + fn)
        fpr = float(fp) / (fp + tn)
        pss = tpr - fpr
        return  [pss, tpr, fpr] 
    if skill == 'hss': #Heidke
        return 2.0 * (tp*tn - fp*fn) / ((tp+fn) * (fn+tn) + (tp+fp) * (fp+tn))
    if skill == 'bss': #Brier Skill Score
        return np.mean((y_true - y_pred) **2)

def plot_confusion_matrix(cm, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Under constructions
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    PARAMS:
        cm: the confusion matrix
        classes: list of unique class labels
        normalize: boolean flag whether to normalize values
        title: figure title
        cmap: colormap scheme
    """
    # View percentages
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color='w' if cm[i, j] > thresh else 'k')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_mtx', bbox_inches="tight")



# Compute the ROC and PR Curves and generate the KS plot
def ks_roc_prc_plot(targets, scores, FIGWIDTH=15, FIGHEIGHT=6, FONTSIZE=14):
    ''' 
    Generate a figure that plots the ROC and PR Curves and the distributions 
    of the TPR and FPR over a set of thresholds. ROC plots the false alarm rate 
    versus the hit rate. The precision-recall curve (PRC) displays recall vs 
    precision
    PARAMS:
        targets: list of true target labels
        scores: list of predicted labels or scores
    RETURNS:
        roc_results: dict of ROC results: {'tpr', 'fpr', 'thresholds', 'AUC'}
        prc_results: dict of PRC results: {'precision', 'recall', 
                                           'thresholds', 'AUC'}
        fig, axs: corresponding handles for the figure and axis
    '''
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(targets, scores)
    auc_roc = auc(fpr, tpr)

    # Compute precision-recall AUC
    precision, recall, thresholds_prc = precision_recall_curve(targets, scores)
    auc_prc = auc(recall, precision)

    roc_results = {'tpr':tpr, 'fpr':fpr, 'thresholds':thresholds, 'auc':auc_roc}
    prc_results = {'precision':precision, 'recall':recall,
                   'thresholds':thresholds_prc, 'auc':auc_prc}
    #thresholds = {'roc_thres':thresholds, 'prc_thres':thresholds_prc}
    #auc_results = {'roc_auc':auc_roc, 'prc_auc':auc_prc}

    # Compute positve fraction
    pos = np.where(targets)[0]
    npos = targets.sum()
    pos_frac = npos / targets.size

    # Generate KS plot
    fig, ax = plt.subplots(1, 3, figsize=(FIGWIDTH,FIGHEIGHT))
    axs = ax.ravel()
    
    ax[0].plot(thresholds, tpr, color='b')
    ax[0].plot(thresholds, fpr, color='r')
    ax[0].plot(thresholds, tpr - fpr, color='g')
    ax[0].invert_xaxis()
    #ax[0].set_aspect('equal', 'box')
    ax[0].set(xlabel='threshold', ylabel='fraction')
    ax[0].legend(['TPR', 'FPR', 'K-S Distance'], fontsize=FONTSIZE)
    
    # Generate ROC Curve plot
    ax[1].plot(fpr, tpr, color='b')
    ax[1].plot([0,1], [0,1], 'r--')
    ax[1].set(xlabel='FPR', ylabel='TPR')
    ax[1].set_aspect('equal', 'box')
    auc_text = ax[1].text(.05, .95, "AUC = %.4f" % auc_roc, 
                          color="k", fontsize=FONTSIZE)
    #print("ROC AUC:", auc_roc)
    
    # Generate precision-recall Curve plot
    ax[2].plot(recall, precision, color='b')
    ax[2].plot([0, 0, 1], [1, pos_frac, pos_frac], 'r--')
    ax[2].set(xlabel='Recall', ylabel='Precision')
    ax[2].set_aspect('equal', 'box')
    auc_prc_text = plt.text(.2, .95, "PR AUC = %.4f" % auc_prc, 
                            color="k", fontsize=FONTSIZE)
    pos_frac_text = plt.text(.2, .85, "%.2f %% pos" % (pos_frac * 100), 
                             color="k", fontsize=FONTSIZE)
    #print("PRC AUC:", auc_prc)

    return roc_results, prc_results, fig, axs

def objective_surface3D(paramX,paramY,z, view_angle=30, view_elev=30):

    fig= plt.figure(figsize=(10,10))
    x,y= np.meshgrid(paramX, paramY)
    assert x.shape==z.shape, 'expected z: %s, but get %s'%(str(x.shape), str(z.shape))
    ax= fig.gca(projection='3d')
    ax.plot_surface(x,y, z, cmap='jet')
    ax.view_init(view_elev,view_angle)
    ax.set_xlabel('thresholds')
    ax.set_ylabel('components')
    ax.set_zlabel('K-S distance')
    #find optimal
    zmax= np.argmax(z)
    optimY= zmax//paramX
    optimX= zmax%paramY
    # ax.scatter(-4.5,7,0.9, edgecolors='k', facecolors='none')
    ax.text(optimX, optimY, z[zmax], 'optimal point (-4.5,7,0.76)')

    return fig, ax








        