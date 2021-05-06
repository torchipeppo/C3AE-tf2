"""
Genera dei grafici in base ai dati salvati nella history di tf.
Usato come script stand-alone a training fatto.
"""

import matplotlib.pyplot as plt
import pickle
import os

def plot_loss(history, name=""):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('{} loss'.format(name), loc='left')
    plt.title("min = {:.3g}".format(min(history['val_loss'])), loc='right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    for i in range(0, len(history['val_loss']), 15):
        plt.annotate(
            text="{:.3g}".format(history['val_loss'][i]), 
            xy=(i, history['val_loss'][i]),
            xytext=(0,10),
            textcoords='offset points'
        )
    plt.show()

def plot_cascade_KLD(history, name=""):
    if "W1_loss" not in history.keys():
        plt.title('ERROR: Name "{}" has no Cascade Module'.format(name))
        plt.show()
        return
    plt.plot(history['W1_loss'])
    plt.plot(history['val_W1_loss'])
    plt.title('{} cascade KLD'.format(name), loc='left')
    plt.title("min = {:.3g}".format(min(history['val_W1_loss'])), loc='right')
    plt.ylabel('cascade KL Divergence')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    for i in range(0, len(history['val_W1_loss']), 15):
        plt.annotate(
            text="{:.3g}".format(history['val_W1_loss'][i]), 
            xy=(i, history['val_W1_loss'][i]),
            xytext=(0,10),
            textcoords='offset points'
        )
    plt.show()

def plot_mae(history, name=""):
    key = 'age_mae'
    val_key = 'val_age_mae'
    if key not in history.keys():
        key = 'mae'
        val_key = 'val_mae'
    plt.plot(history[key])
    plt.plot(history[val_key])
    plt.title('{} age MAE'.format(name), loc='left')
    plt.title("min = {:.3g}".format(min(history[val_key])), loc='right')
    plt.ylabel('age MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    for i in range(0, len(history[val_key]), 15):
        plt.annotate(
            text="{:.3g}".format(history[val_key][i]), 
            xy=(i, history[val_key][i]),
            xytext=(0,10),
            textcoords='offset points'
        )
    plt.show()

import path_constants

if __name__=="__main__":
    with open(os.path.join(path_constants.MODEL_HISTORY_SAVE, "history_wiki_100epochs_fullablation.hist"), "rb") as f:
        history = pickle.load(f)
    name="Ablation (no context and no cascade)"
    plot_loss(history, name)
    plot_cascade_KLD(history, name)
    plot_mae(history, name)