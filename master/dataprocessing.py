'''
questo modulo andrà chiamato una volta sola
e trasforma il dataset di immagini
in dataframe con tutte le info
'''

import pandas as pd
import cv2
import pickle
import re
import os

PATH_COL = "path"
AGE_COL = "age"
IMAGE_COL = "image"
CROP_BOXES_COL = "crop_boxes"

PRE_COLS = (PATH_COL, AGE_COL)
COLS = (IMAGE_COL, AGE_COL, CROP_BOXES_COL)

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
    width, height = image.shape[0:2]   # il 2 è escluso, quindi prende [0] e [1]
    assert width<height, "sei sicuro che sia una faccia?"
    # posizioniamo i quadrati un po' sopra il punto di mezzo
    square_middle = (
        int(witdh/2), 
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
    status, encoded_img = cv2.imencode(".jpg", image)
    row[IMAGE_COL] = encoded_img.tostring()
    row[CROP_BOXES_COL] = crop_boxes.dumps()

# salva il dataset in chunk da N righe
def save_dataset(dataset, chunk_size=5000):
    chunk_start=0
    print(dataset)

def process_fgnet(dataset_dir):
    early_dataset = pd.DataFrame(columns = PRE_COLS)
    for root, dirs, files in os.walk(dataset_dir):
        for fname in files:
            found = re.findall(r'[\d]+', fname)
            assert len(found)==2, "Error"
            # id = int(found[0])
            age = int(found[1])
            path = os.path.abspath(os.path.join(root, fname))
            early_dataset.append({PATH_COL:path, AGE_COL:age}, ignore_index=True)
    
    true_dataset = make_true_dataset(early_dataset)

    save_dataset(true_dataset)

if __name__=="__main__":
    process_fgnet("..\dataset\FGNET\FGNET\images")





