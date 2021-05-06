"""
Verifica che i modelli siano costruiti correttamente
tentando di invocarli e assicurandosi che non lancino eccezioni.
"""

import numpy as np
import pandas as pd

from master import dataprocessing, image_manipulation, model
import mtcnn     # EXTERNAL CODE, see readme in mtcnn
MTCNN = mtcnn.mtcnn.MTCNN

MTCNN_DETECT = MTCNN()

def mrt_main(image_path, seed=14383421):
    # creiamo quattro modelli
    model_full = model.full_context_model(name="model_full")
    model_no_context = model.full_context_model(name="model_no_context", ablate_context=True)
    model_no_cascade = model.full_context_model(name="model_no_cascade", ablate_cascade=True)
    model_no_nothing = model.full_context_model(name="model_no_nothing", ablate_context=True, ablate_cascade=True)

    # possiamo riutilizzare le funzioni scritte in precedenza,
    # ma dovremo fare delle mosse aggiuntive per adattarci
    # alla loro interfaccia
    pseudo_row = pd.Series({"name": 0, "path": image_path})
    pseudo_row = dataprocessing.read_image_and_process(pseudo_row, detector=MTCNN_DETECT)

    pseudo_rwi = [0, pseudo_row]
    crops = image_manipulation.get_image_crops(pseudo_rwi, seed, False)

    # i modelli Keras vogliono necessariamente una batch, anche per un solo elemento
    pseudo_batch = crops[np.newaxis, ...]

    three_inputs = [pseudo_batch[:,0], pseudo_batch[:,1], pseudo_batch[:,2]]

    print("-------- MODELLO CON TUTTO --------")
    # invochiamo il modello per avere la predizione
    results_pseudo_batches = model_full(three_inputs)
    # recuperiamo l'unico risultato
    age_pseudo_batch = results_pseudo_batches[0]
    age = age_pseudo_batch[0]
    print(age)

    print()
    print("-------- MODELLO SENZA CONTEXT --------")
    # invochiamo il modello per avere la predizione
    results_pseudo_batches = model_no_context(three_inputs[0:1])    # questa notazione ci assicura di avere comunque una lista
    # recuperiamo l'unico risultato
    age_pseudo_batch = results_pseudo_batches[0]
    age = age_pseudo_batch[0]
    print(age)

    print()
    print("-------- MODELLO SENZA CASCADE --------")
    # invochiamo il modello per avere la predizione
    results_pseudo_batches = model_no_cascade(three_inputs)
    # recuperiamo l'unico risultato
    age_pseudo_batch = results_pseudo_batches[0]
    age = age_pseudo_batch[0]
    print(age)

    print()
    print("-------- MODELLO SENZA NIENTE --------")
    # invochiamo il modello per avere la predizione
    results_pseudo_batches = model_no_nothing(three_inputs[0:1])    # questa notazione ci assicura di avere comunque una lista
    # recuperiamo l'unico risultato
    age_pseudo_batch = results_pseudo_batches[0]
    age = age_pseudo_batch[0]
    print(age)