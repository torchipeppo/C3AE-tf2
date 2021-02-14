'''
questo modulo andrà chiamato una volta sola
e trasforma il dataset di immagini
in dataframe con tutte le info
'''

import pandas as pd
import numpy as np
from cv2 import cv2   # for visual studio code
import pickle
import re
import os

import path_constants

PATH_COL = "path"
AGE_COL = "age"
IMAGE_COL = "image"
CROP_BOXES_COL = "crop_boxes"

PRE_COLS = [PATH_COL, AGE_COL]
COLS = [IMAGE_COL, AGE_COL, CROP_BOXES_COL]

def make_true_dataset(early_dataset):
    # usiamo una funzione ausiliaria per leggere ed elaborare tutte le immagini
    dataset = early_dataset.apply(
        # usiamo questa notazione estesa per renderci più facile aggiungere argomenti alla funzione in futuro
        lambda row : read_image_and_process(row),
        axis=1
    )

    # filtriamo le età che non hanno senso
    dataset = dataset[(dataset.age>=0) & (dataset.age<=120)]

    # selezioniamo le colonne utili al training
    dataset = dataset[COLS]

    return dataset

# come prima versione tagliamo le immagini con quadrati fissi
# TODO magari fa pena, ma intanto prototipiamo così
def fixed_squares(image):
    # height == numero di righe
    # width == numero di colonne
    height, width = image.shape[0:2]   # il 2 è escluso, quindi prende [0] e [1]
    assert width<=height, "sei sicuro che sia una faccia?"
    # posizioniamo i quadrati un po' sopra il punto di mezzo
    square_middle = (
        int(width/2), 
        int(height/2 - (height-width)/4)
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

def read_image_and_process(row):
    image_path = row.path
    print("Elaborando: {}".format(image_path))
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # E ADESSO GENERIAMO I QUADRATI (fissi)
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

def process_fgnet(dataset_dir, overwrite=False):
    name = "fgnet"

    # controllo file esistente
    if file_search_in_folder(name, path_constants.FGNET_OUTPUT) and not overwrite:
        print("Un file chiamato "+name+" esiste già")
        return

    # early_dataset = pd.DataFrame(columns = PRE_COLS)
    paths = []
    ages = []
    for root, __dirs, files in os.walk(dataset_dir):
        for fname in files:
            found = re.findall(r'[\d]+', fname)
            assert len(found)==2, "Error"
            # id = int(found[0])
            age = int(found[1])
            path = os.path.abspath(os.path.join(root, fname))
            # early_dataset.append({PATH_COL:path, AGE_COL:age}, ignore_index=True)
            paths.append(path)
            ages.append(age)
    early_dataset = pd.DataFrame({PATH_COL: paths, AGE_COL: ages})

    true_dataset = make_true_dataset(early_dataset)

    save_dataset(
        true_dataset,
        path_constants.FGNET_OUTPUT,
        name
    )

if __name__=="__main__":
    process_fgnet(path_constants.FGNET_IMAGES)





