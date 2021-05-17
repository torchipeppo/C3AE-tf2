"""
Permette di predire l'età di una singola foto data in input, 
restituendo la stessa immagine modificata con l'aggiunta delle boxes 
e dell'età 
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from cv2 import cv2
import pickle
import os

from master import dataprocessing, image_manipulation, model
from utils import path_constants
import mtcnn     # EXTERNAL CODE, see readme in mtcnn
MTCNN = mtcnn.mtcnn.MTCNN

MTCNN_DETECT = MTCNN()

def sp_main(image_path, model_path, silent=False, seed=14383421):
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

    # carichiamo il modello
    model = keras.models.load_model(model_path, custom_objects={'tf': tf})

    # invochiamo il modello per avere la predizione
    results_pseudo_batches = model(three_inputs)
    # recuperiamo l'unico risultato
    age_pseudo_batch = results_pseudo_batches[0]
    distr_pseudo_batch = results_pseudo_batches[1]
    age = age_pseudo_batch[0]
    distr = distr_pseudo_batch[0]

    image = np.frombuffer(pseudo_row.image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    crop_boxes = pickle.loads(pseudo_row.crop_boxes, encoding="bytes")
    min_size = min(image.shape[0], image.shape[1])
    for i in range(3):
        # print(crop_boxes)
        pt1 = crop_boxes[i][0]
        pt2 = crop_boxes[i][1]
        x1,y1=pt1
        x2,y2=pt2
        pt1=(x1,y1)
        pt2=(x2,y2)
        if i==0:
            color=(0,0,255)
        elif i==1:
            color=(0,255,0)
        else:
            color=(255,0,0)
        thickness = max(2, int(0.005*min_size))
        image = cv2.rectangle(image, pt1, pt2, color, thickness)
    
    age = int(np.round(age.numpy()[0]))
    
    font_scale = max(1, int(0.0025*min_size))
    font_thickness = max(3, int(0.005*min_size))
    text_offset = min(30, int(0.025*min_size))
    text_size, _ = cv2.getTextSize(str(age), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    _, text_height = text_size

    image = cv2.putText(
        image, 
        str(age), 
        (text_offset, text_offset+text_height), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale,
        (0,0,0), 
        font_thickness
    )

    # print(age)
    # print(distr)
    cv2.imwrite(image_path+".squares.jpg", image)