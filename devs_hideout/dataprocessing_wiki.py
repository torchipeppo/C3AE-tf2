'''
questo modulo andrà chiamato una volta sola
e trasforma il dataset di immagini
in dataframe con tutte le info
'''

import pandas as pd
import numpy as np
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
#from mtcnn.mtcnn.mtcnn import MTCNN     # EXTERNAL CODE, see readme in mtcnn
import mtcnn
MTCNN = mtcnn.mtcnn.MTCNN

PATH_COL = "path"
AGE_COL = "age"
IMAGE_COL = "image"
CROP_BOXES_COL = "crop_boxes"

PRE_COLS = [PATH_COL, AGE_COL]
COLS = [IMAGE_COL, AGE_COL, CROP_BOXES_COL]

MTCNN_DETECT = MTCNN() #min_face_size=50?

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

def read_image_and_process(row, detector):
    
    if row.name%600==1:
        print(int(row.name)/62000)
    image_path = row.path
    #print("Elaborando: {}".format(image_path))
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # E ADESSO GENERIAMO I QUADRATI (Qui non sono fissi ma generati grazie alla face detection)
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

def process_wiki(dataset_dir, overwrite=False):
    name = "wiki"
    # controllo file esistente
    if file_search_in_folder(name, path_constants.WIKI_PICKLE) and not overwrite:
        print("Un file chiamato "+name+" esiste già")
        return

    paths = []
    ages = []
    
    for root, __dirs, files in os.walk(dataset_dir):
        for fname in files:
            if fname=="wiki.mat": continue # skippo
            found = re.findall(r'[\d]+', fname)
            assert len(found)>=5, "Error"+fname
            year_of_birth = int(found[1])
            year_taken = int(found[-1])
            age = year_taken - year_of_birth
            #assert age>=0, "Hai fotografato un feto?"
            ages.append(age)
            path = os.path.abspath(os.path.join(root, fname))
            paths.append(path)
    #print(ages)
            
    early_dataset = pd.DataFrame({PATH_COL: paths, AGE_COL: ages})

    true_dataset = make_true_dataset(early_dataset)

    save_dataset(
        true_dataset,
        path_constants.WIKI_PICKLE,
        name
    )

if __name__=="__main__":

    #senza questa riga non sembra funzionare: da approfondire
    #os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 

    with tf.device('/cpu:0'):
        process_wiki(path_constants.WIKI_IMAGES)





