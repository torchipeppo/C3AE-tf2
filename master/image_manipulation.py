import numpy as np
import cv2
import pickle
import random

from utils import path_constants

def random_erasing(image, area_ratio_bounds=(0.06,0.10), aspect_ratio_bounds=(0.5,2)):
    # copiamo l'immagine e recuperiamo le dimensioni
    image = image.copy()
    height, width = image.shape[:-1]
    # generiamo a caso un rettangolo da ritagliare.
    # campioniamo prima la sua area in termini del rapporto con l'area dell'immagine:
    area_ratio = np.random.uniform(*area_ratio_bounds)
    erase_area = (width*height)*area_ratio
    # le dimensioni le determiniamo indirettamente campionando l'aspect ratio del rettangolo:
    aspect_ratio = np.random.uniform(*aspect_ratio_bounds)
    erase_width = int(np.round(np.sqrt(erase_area * aspect_ratio)))
    erase_height = int(np.round(np.sqrt(erase_area * 1 / aspect_ratio)))
    # ora non resta che campionare uno dei vertici del rettangolo
    x0 = random.randint(0, height-erase_height)
    y0 = random.randint(0, width-erase_width)
    # ...e l'altro è determinato dalle dimensioni
    x1 = min(height, x0 + erase_height)
    y1 = min(width, y0 + erase_width)
    # infine procediamo a cancellare davvero
    image[x0:x1, y0:y1, :] = np.random.random_integers(0, 256, (x1-x0,y1-y0,3))
    return image

def do_some_augmentation(
    image,
    probability_of_one_transformation=0.5,
    contrast_bounds=(0.5, 2.5),
    brightness_bounds=(-64, 64),
    rotation_bounds=(-15, 15)
):
    image=image.copy()
    # cambio contrasto/luminosità
    if np.random.rand() < probability_of_one_transformation:
        contrast = np.random.uniform(*contrast_bounds)
        brightness = np.random.uniform(*brightness_bounds)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    # rotazione
    if np.random.rand() < probability_of_one_transformation:
        height, width = image.shape[:-1]
        rotation = np.random.uniform(*rotation_bounds)
        rotation_matrix = cv2.getRotationMatrix2D((height//2, width//2), rotation, 1)
        image = cv2.warpAffine(image, rotation_matrix, (height,width))
    # flipping verticale
    if np.random.rand() < probability_of_one_transformation:
        image = cv2.flip(image, 1)
    # TODO shift dei quadrati? se sì va implementata in get_image_crops
    # TODO temperatura (super flex)? se sì, possibile fonte: https://www.askaswiss.com/2016/02/how-to-manipulate-color-temperature-opencv-python.html
    return image

def get_image_crops(row_with_index, seed, augment, nn_input_shape=(64,64), padding=250, erasing_probability=0.3):
    __index, row = row_with_index[0], row_with_index[1]
    image = np.frombuffer(row.image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if augment:
        if np.random.rand() < erasing_probability:
            image = random_erasing(image)

    # aggiungiamo del padding per non morire in caso di boundingbox out-of-range
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_DEFAULT)

    crops = []
    for box in pickle.loads(row.crop_boxes, encoding="bytes"):  # invece di np.loads
        h0, w0 = np.int64(box[0])   # conversione np da float a int
        h1, w1 = np.int64(box[1])
        crops.append(cv2.resize(padded_image[h0+padding:h1+padding, w0+padding:w1+padding, :], nn_input_shape))
    
    if augment:
        crops = [do_some_augmentation(img) for img in crops]

    # return crops
    return np.array(crops)