import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
import math
import pickle
import json
import sqlite3
from tensorflow import keras


##############################################################
# Makes a roc curve of two lists: a=predictions, b=truth
# Does not account for probabilities, only guesses
def _ROC(a, b):
    if len(a) != len(b):
        raise Exception('The two lists are not of equal length.')

    c, w = np.zeros_like(b), np.zeros_like(b)
    ccounter, wcounter = 1, 1
    
    if a[0] == b[0]:
        c[0] = 1
    else:
        w[0] = 1
        
    for i in range(len(a)-1):
        if a[i] == b[i]:
            c[i+1] = ccounter
            w[i+1] = w[i]
            ccounter += 1
        else:
            w[i+1] = wcounter
            c[i+1] = c[i]
            wcounter += 1
            
    if c[-1] == 0:
        print('Zero correct predictions.')
    else:
        c = c/c[-1]
    if w[-1] == 0:
        print('Zero wrong predictions.')
    else:
        w = w/w[-1]

#    c = c/c[-1]
#    w = w/w[-1]
    
    #plot
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(w,c)
    ax.plot(np.linspace(0,1,len(a)) ,np.linspace(0,1,len(a)), c='k', linestyle='--')
    ax.set_xlabel('Wrong')
    ax.set_ylabel('Correct')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    return c, w # scale to percentage


def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)


def deb_tcn_0(timesteps, input_dim, output_dim):
    i = tf.keras.Input(batch_shape = inputshape)
    o = TCN(nb_filters = 32, activation = 'sigmoid')(i)
    o = TCN(nb_filters = 64, activation = 'sigmoid')(o)
    o = tf.keras.layers.Dense(output_dim, activation='sigmoid')(o)
    model = tf.keras.models.Model(inputs=[i], outputs=[o])
    return model
      
def open_json_file(path):
    with open(str(path), "r") as f:
        output = json.load(f)
    return output

    
################ AppStat ROC #############################

def calc_ROC(hist1, hist2):

    # hist1 is signal, hist2 is background

    # first we extract the entries (y values) and the edges of the histograms
    y_sig, x_sig_edges, _ = hist1
    y_bkg, x_bkg_edges, _ = hist2

    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges):

        # extract the center positions (x values) of the bins (doesn't matter if we use signal or background because they are equal)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])

        # calculate the integral (sum) of the signal and background
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()

        # initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR).
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()

        # loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin
        for i, x in enumerate(x_centers):

            # the cut mask
            cut = (x_centers < x)

            # true positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / float(TP + FN)                    # True positive rate

            # true negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / float(FP + TN)                     # False positive rate

        return FPR, TPR

    else:
        AssertionError("Signal and Background histograms have different bins and ranges")

def plot_ROC(FPR, TPR, labels=None, colors=None, figsize=(12,6), ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)


    for i in np.arange(len(FPR)):

        kwargs = {}
        if colors is not None:
            kwargs['color'] = colors[i]
        if labels is not None:
            kwargs['label'] = labels[i]

        ax.plot(FPR[i], TPR[i], **kwargs)


    ax.plot([0,1], [0,1], 'k--')
    if labels is not None:
        ax.legend()
    ax.set(xlabel='False Positive Rate (Background Efficiency)', ylabel='True Positive Rate (Signal Efficiency)', xlim=(0, 1), ylim=(0, 1))

    return ax

def AppStat_ROCmaker(prediction, target):
    pred_hist = plt.hist(prediction)
    targ_hist = plt.hist(target)
    fpr, tpr = calc_ROC(pred_hist, targ_hist)
    plot_ROC([fpr,], [tpr,])
    

################################################################    

def ROC_plot(preds, targs, save_plot = False):
    N_preds = len(targs)
    
    guesses = np.zeros((N_preds,1))
    for i in range(N_preds):
        guesses[i] = int(list(preds[i]).index(np.max(list(preds)[i])))
        
    
    test, pred = targs, preds[:,1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = metrics.roc_curve(test, pred)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    auc = metrics.roc_auc_score(test, pred)
    textboxparams = dict(boxstyle='round', facecolor='grey', alpha=0.2)
    fig, ax = plt.subplots(figsize = (12,8))
    ax.plot(fpr[1], tpr[1])   
    text_dict = {'AUC': f'{auc:.4f}',
              'Accuracy': (len(test)-sum(abs(pred-test)))/len(test),
              'Truth':      str(np.round((np.count_nonzero(targs)/len(targs))*100,3))+'% stopped muons',
              'Prediction': str(np.round((np.count_nonzero(guesses)/len(guesses))*100,3))+'% stopped muons'}
   
    text = nice_string_output(text_dict, extra_spacing=2, decimals=4)
    add_text_to_ax(0.4, 0.3, text, ax, fontsize=20)
    
    ax.plot(np.linspace(0,1,len(fpr)), np.linspace(0,1,len(fpr)) , '--', c='k')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    plt.show()
    if save_plot:
            fig.savefig(pred_file[:-9] + 'ROC'+'.pdf', dpi=600, format='pdf', bbox_inches='tight')

            
def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None