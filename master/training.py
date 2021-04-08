import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from master import model as model_module
import tensorflow.keras as keras
import cv2
import os
from datetime import datetime
import pickle
import math
from master.dataprocessing import COLS
import tensorflow as tf
import random

from utils import path_constants
from master import image_manipulation

def age2twopoint(age, categories, interval):
    twopoint_vector = [0 for x in range(categories)]
    right_prob = age % interval * 1.0 / interval
    left_prob = 1 - right_prob
    left_index = age // interval
    right_index = left_index+1
    twopoint_vector[left_index] = left_prob
    if right_index<categories:
        twopoint_vector[right_index] = right_prob
    
    return np.array(twopoint_vector)

def datagen(dataframe, batch_size, seed, categories, interval, augment, ablate_context, ablate_cascade):
    dataframe = dataframe.reset_index(drop=True)
    df_len = len(dataframe)
    while True:
        # inizio ciclo lungo la dataframe
        permutated_indices = np.random.permutation(df_len)
        start = 0
        while start+batch_size < df_len:
            index_batch = list(permutated_indices[start : start+batch_size])
            row_batch = dataframe.iloc[index_batch]
            image_batch = np.array([
                image_manipulation.get_image_crops(rwi, seed, augment) for rwi in row_batch.iterrows()
            ])
            crops_batch_list = [image_batch[:,0], image_batch[:,1], image_batch[:,2]]
            if ablate_context:
                # se usiamo un modello senza context, deve prendere una sola immagine come input
                crops_batch_list = crops_batch_list[0:1]   # questa notazione prende solo il primo elemento, ma dà una lista anziché solo quell'elemento
            age_array = row_batch.age.to_numpy()
            values = [
                age_array,
                np.array([age2twopoint(a, categories, interval) for a in age_array])
            ]
            if ablate_cascade:
                # se usiamo un modello senza cascade, non esiste la rappresentazione two-point,
                # quindi non dobbiamo dare la groundtruth corrispondente
                values = values[0:1]   # vedi sopra
            yield crops_batch_list, values
            start += batch_size

def do_train(
    dataset, 
    train_split=0.888, 
    batch_size=50,
    lr=0.002,
    loss_weight_factor=10,
    seed=14383421,
    bins=10,
    epochs=10,   # 250
    augment=True,
    ablate_context=False,
    ablate_cascade=False,
    pretrained_model_path=None
):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    categories = bins+2
    interval = int(math.ceil(100.0/bins))

    trainset, validset = train_test_split(
        dataset, train_size=train_split
    )
    train_gen = datagen(trainset, batch_size, seed, categories, interval, augment, ablate_context, ablate_cascade)
    valid_gen = datagen(validset, batch_size, seed, categories, interval, False, ablate_context, ablate_cascade)

    if not pretrained_model_path:
        # se non specifichiamo un modello pretrainato, creiamo il modello da zero
        model = model_module.full_context_model(
            bottleneck_dim=categories,
            ablate_context=ablate_context,
            ablate_cascade=ablate_cascade
        )
    else:
        # altrimenti carichiamo il modello pretrainato
        model = keras.models.load_model(pretrained_model_path, custom_objects={'tf': tf})

    opti = keras.optimizers.Adam(lr=lr)

    losses = ["mae", "kullback_leibler_divergence"]
    loss_weights = [1,loss_weight_factor]
    
    if ablate_cascade:
        # se usiamo un modello senza cascade, non esiste la rappresentazione two-point,
        # quindi non dobbiamo considerare la perdita
        losses = losses[0:1]   # questa notazione prende solo il primo elemento, ma dà una lista anziché solo quell'elemento
        loss_weights = None

    model.compile(
        optimizer=opti,
        loss=losses,
        metrics={"age":"mae"},
        loss_weights=loss_weights
    )

    history = model.fit(    # anziché fit_generator
        train_gen,
        steps_per_epoch = len(trainset)/batch_size,
        epochs=epochs,   # 250
        validation_data=valid_gen,
        validation_steps = len(validset)/batch_size*3
    )

    now = datetime.now()
    model_fname = "model_{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}.h5".format(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second
    )
    model_path = os.path.join(path_constants.MODEL_HISTORY_SAVE, model_fname)
    model.save(model_path)
    history_fname = "history_{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}.hist".format(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second
    )
    history_path = os.path.join(path_constants.MODEL_HISTORY_SAVE, history_fname)
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)

def train_main(dataset_pickle_path, epochs, ablation, pretrained_model_path):
    #senza questa riga non sembra funzionare: da approfondire
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'    

    if tf.test.gpu_device_name(): 
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("TF will be used with CPU")

    files = os.listdir(dataset_pickle_path)
    frames=[]
    for fname in files:
        if ".pkl" not in fname: continue   # skippa ciò che non è un pickle
        path = os.path.join(dataset_pickle_path, fname)
        frame = pd.read_pickle(path)
        frames.append(frame)
    dataset = pd.concat(list(frames), ignore_index=True)
    # filtriamo le età che non hanno senso
    dataset = dataset[(dataset.age>=0) & (dataset.age<120)]
    dataset = dataset.dropna()

    print(f"Caricato dataset da {dataset.shape[0]} immagini")

    print()
    print("################ Normal ###################")
    print()
    do_train(dataset, epochs=epochs, pretrained_model_path=pretrained_model_path)
    if ablation:
        print()
        print("################ No Augmentation ###################")
        print()
        do_train(dataset, epochs=epochs, augment=False)
        print()
        print("################# No Context ###################")
        print()
        do_train(dataset, epochs=epochs, augment=True, ablate_context=True)
        print()
        print("#################### No Cascade ##############")
        print()
        do_train(dataset, epochs=epochs, augment=True, ablate_cascade=True)
        print()
        print("############### Full Ablation ###############")
        print()
        do_train(dataset, epochs=epochs, augment=True, ablate_context=True, ablate_cascade=True)
        pass   # TMCH per quando l'if è tutto commentato
