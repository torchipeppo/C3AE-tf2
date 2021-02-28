'''
questo modulo andrà chiamato una volta sola
e trasforma il dataset di immagini
in dataframe con tutte le info
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from cv2 import cv2   # for visual studio code
import pickle
import re
import os
from datetime import datetime
import scipy.io

from utils import path_constants

import sys
sys.path.append(path_constants.REPO_ROOT)
import mtcnn     # EXTERNAL CODE, see readme in mtcnn
MTCNN = mtcnn.mtcnn.MTCNN

PATH_COL = "path"
AGE_COL = "age"
IMAGE_COL = "image"
CROP_BOXES_COL = "crop_boxes"

PRE_COLS = [PATH_COL, AGE_COL]
COLS = [IMAGE_COL, AGE_COL, CROP_BOXES_COL]

MTCNN_DETECT = MTCNN() #min_face_size=50?  (TODO cancellare)

def make_true_dataset(early_dataset):
    # usiamo una funzione ausiliaria per leggere ed elaborare tutte le immagini
    dataset = early_dataset.apply(
        # usiamo questa notazione estesa per renderci più facile aggiungere argomenti alla funzione in futuro
        lambda row : read_image_and_process(row, MTCNN_DETECT),
        axis=1
    )

    # filtriamo le età che non hanno senso
    dataset = dataset[(dataset.age>=0) & (dataset.age<=120)]

    # selezioniamo le colonne utili al training
    dataset = dataset[COLS]

    return dataset

# come prima versione tagliamo le immagini con quadrati fissi
# molto subottimale, ma è stato utile per i primi prototipi
def fixed_squares(image):
    # height == numero di righe
    # width == numero di colonne
    height, width = image.shape[0:2]   # il 2 è escluso, quindi prende [0] e [1]
    assert width<=height, "sei sicuro che sia una faccia?"
    # posizioniamo i quadrati un po' sopra il punto di mezzo
    square_middle = (
        int(height/2 - (height-width)/4),
        int(width/2)
    )
    side0 = width   # dove "side" vuol dire "lato del quadrato"
    side1 = int(0.8*side0)
    side2 = int(0.6*side0)
    def box_from_middle_and_side(middle, side):
        return [
            (middle[0]-side/2, middle[1]-side/2),
            (middle[0]+side/2, middle[1]+side/2)
        ]
    return np.array([
        box_from_middle_and_side(square_middle, side0), 
        box_from_middle_and_side(square_middle, side1), 
        box_from_middle_and_side(square_middle, side2), 
    ])

def generate_the_three_boundingboxes(box, keypoints):
    # genera bounding boxes
    x1, y1, x2, y2 = map(int, [box[0][0], box[0][1], box[1][0], box[1][1]])
    w, h = x2 - x1, y2 - y1
    nose_x, nose_y = keypoints["nose"]
    w_h_margin = abs(w - h)
    top_to_nose = nose_y - y1
    return np.array([
        [(x1 - w_h_margin, y1 - w_h_margin), (x2 + w_h_margin, y2 + w_h_margin)],                        # esterna
        [(nose_x - top_to_nose, nose_y - top_to_nose), (nose_x + top_to_nose, nose_y + top_to_nose)],    # di mezzo
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]                                 # interna
    ])

# conversione del formato delle bounding box
def convert_box_ipazc_to_ours(ipazc_box):
    [x,y,w,h] = ipazc_box
    return [
        (x  ,y  ),
        (x+w,y+h)
    ]

def read_image_and_process(row, detector=None):
    # stampa di stato (se possibile)
    if row.name:
        if row.name%600==1:
            print(int(row.name)/62000)

    image_path = row.path
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # E ADESSO GENERIAMO I QUADRATI
    if detector:
        # Quadrati generati grazie alla face detection
        detect_results = detector.detect_faces(image)    # una lista di dizionari, ciascuno contenente box, confidence, keypoints
        #se troviamo nessuna o tante facce
        if len(detect_results)!=1:
            # scartiamo questa riga (delegando lo scarto vero e proprio al resto di make_true_dataset)
            row[AGE_COL] = -190405059
            return row
        box = convert_box_ipazc_to_ours(detect_results[0]["box"])
        keypoints = detect_results[0]["keypoints"]
        # MEMENTO: se dovessimo usare un dataset che ha bisogno di allineare le facce, questo è un buon punto per farlo
        #          in tal caso, ricordarsi di ri-detectare la faccia subito dopo
        crop_boxes = generate_the_three_boundingboxes(box, keypoints)
    else:
        # Quadrati fissi, vicino al centro dell'immagine.
        # molto subottimale, ma è stato utile per i primi prototipi.
        crop_boxes = fixed_squares(image)
    __status, encoded_img = cv2.imencode(".jpg", image)
    row[IMAGE_COL] = encoded_img.tobytes()
    row[CROP_BOXES_COL] = crop_boxes.dumps()

    return row

# salva il dataset in chunk da N righe
def save_dataset(dataset, output_dir, name, chunk_size=5000):
    dataset = dataset.reset_index(drop=True)
    chunk_start=0
    while chunk_start < len(dataset):
        save_path = os.path.join(
            output_dir,
            name+"_"+str(int(chunk_start/chunk_size))+".pkl"
        )
        chunk = dataset[chunk_start : chunk_start+chunk_size].copy().reset_index()
        chunk.to_pickle(save_path)
        chunk_start+=chunk_size

def file_search_in_folder(name, dir):
    files = os.listdir(dir)
    for f in files:
        if name in f:
            return True
    return False

def read_early_dataset_fgnet(name="fgnet"):
    dataset_dir = path_constants.IMAGES[name]
    paths = []
    ages = []
    for root, __dirs, files in os.walk(dataset_dir):
        for fname in files:
            found = re.findall(r'[\d]+', fname)
            assert len(found)==2, "Error"
            # id = int(found[0])
            age = int(found[1])
            path = os.path.abspath(os.path.join(root, fname))
            paths.append(path)
            ages.append(age)
    return pd.DataFrame({PATH_COL: paths, AGE_COL: ages})

def read_early_dataset_wiki(name="wiki"):
    dataset_dir = path_constants.IMAGES[name]
    paths = []
    ages = []
    for root, __dirs, files in os.walk(dataset_dir):
        for fname in files:
            if ".jpg" not in fname: continue   # skippa ciò che non è un'immagine
            found = re.findall(r'[\d]+', fname)
            assert len(found)>=5, "Nome immagine malformato: "+fname
            year_of_birth = int(found[1])
            year_taken = int(found[-1])
            age = year_taken - year_of_birth
            #assert age>=0, "Hai fotografato un feto?"
            ages.append(age)
            path = os.path.abspath(os.path.join(root, fname))
            paths.append(path)
    return pd.DataFrame({PATH_COL: paths, AGE_COL: ages})

def process_dataset(name, overwrite=False):
    # controllo file esistente
    if file_search_in_folder(name, path_constants.PICKLES[name]) and not overwrite:
        print("Un file chiamato "+name+" esiste già")
        return
    
    if name=="fgnet":
        early_dataset = read_early_dataset_fgnet(name)
    elif name=="wiki":
        early_dataset = read_early_dataset_wiki(name)

    true_dataset = make_true_dataset(early_dataset)

    save_dataset(
        true_dataset,
        path_constants.PICKLES[name],
        name
    )

if __name__=="__main__":
    # process_fgnet(path_constants.FGNET_IMAGES)

    # with tf.device('/cpu:0'):
        # process_wiki(path_constants.WIKI_IMAGES)
    
    with tf.device('/cpu:0'):
        process_dataset("wiki")





