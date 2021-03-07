import matplotlib.pyplot as plt
import pickle
import os

def plot_loss(history, name=""):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('{} loss'.format(name))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def plot_cascade_KLD(history, name=""):
    plt.plot(history['W1_loss'])
    plt.plot(history['val_W1_loss'])
    plt.title('{} cascade KL Divergence'.format(name))
    plt.ylabel('cascade KL Divergence')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

def plot_mae(history, name=""):
    plt.plot(history['age_mae'])
    plt.plot(history['val_age_mae'])
    plt.title('{} age MAE'.format(name))
    plt.ylabel('age MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

import path_constants

if __name__=="__main__":
    with open(os.path.join(path_constants.MODEL_HISTORY_SAVE, "history_wiki_100epochs_fullmodel_augmentation.hist"), "rb") as f:
        history = pickle.load(f)
    plot_loss(history)
    plot_cascade_KLD(history)
    plot_mae(history)