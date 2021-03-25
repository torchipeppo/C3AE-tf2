import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import sklearn
import random
import math
import os
from master import training    # per riutilizzare datagen
from utils import path_constants

def concatenate_batch_results(result_list):
    restructured = [ [elem] for elem in result_list[0]]
    for pair in result_list[1:]:
        for i in range(len(restructured)):
            restructured[i].append(pair[i])

    results = [ np.concatenate(l,axis=0) for l in restructured ]

    return results

def do_test(
    dataset,
    model_path,
    batch_size=50,
    seed=14383421,
    bins=10,
    ablate_context=False,
    ablate_cascade=False
):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    categories = bins+2
    interval = int(math.ceil(100.0/bins))
    test_gen = training.datagen(dataset, batch_size, seed, categories, interval, False, ablate_context, ablate_cascade)

    model = keras.models.load_model(model_path, custom_objects={'tf': tf})

    dataset_len = dataset.shape[0]
    # predictions = model.predict(test_gen, verbose=1, steps=dataset_len//batch_size)
    preds_list = []
    gt_list = []
    step = 0
    for x,y in test_gen:
        if not step<dataset_len//batch_size:
            break
        preds_list.append(model.predict(x))
        gt_list.append(y)
        step+=1
    predictions = concatenate_batch_results(preds_list)
    groundtruths = concatenate_batch_results(gt_list)

    return predictions, groundtruths

def test_main(dataset_pickle_path, model_path):
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

    preds, gts = do_test(dataset, model_path)

    mae = sklearn.metrics.mean_absolute_error(gts[0], preds[0])
    print("mae =",mae)

    kl_loss = keras.losses.KLDivergence()
    kld = kl_loss(gts[1], preds[1]).numpy()
    print("kld =",kld)
