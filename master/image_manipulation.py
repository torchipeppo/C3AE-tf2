import numpy as np
import cv2
import pickle
import random
import scipy.interpolate

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
    # cambio temperatura
    if np.random.rand() < probability_of_one_transformation:
        image = random_temperature(img)
    return image

def box_random_shift(box, relative_shift_bounds=(-0.1, 0.1)):
    height = np.abs(box[0][0] - box[1][0])
    width = np.abs(box[0][1] - box[1][1])
    relative_shift_vert = np.random.uniform(*relative_shift_bounds)
    relative_shift_horiz = np.random.uniform(*relative_shift_bounds)
    shift_vert = height*relative_shift_vert
    shift_horiz = width*relative_shift_horiz
    return [
        (box[0][0]+shift_vert, box[0][1]+shift_horiz),
        (box[1][0]+shift_vert, box[1][1]+shift_horiz)
    ]

def get_image_crops(row_with_index, seed, augment, nn_input_shape=(64,64), padding=250, erasing_probability=0.3, other_transf_prob_each=0.5):
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
        # augmentation: shift dei quadrati. va fatta qui perché è l'unico punto in cui ABBIAMO i quadrati.
        if augment:
            if np.random.rand() < other_transf_prob_each:
                box = box_random_shift(box)
        h0, w0 = np.int64(box[0])   # conversione np da float a int
        h1, w1 = np.int64(box[1])
        crops.append(cv2.resize(padded_image[h0+padding:h1+padding, w0+padding:w1+padding, :], nn_input_shape))
    
    if augment:
        crops = [do_some_augmentation(img, other_transf_prob_each) for img in crops]

    # return crops
    return np.array(crops)


# The following two functions are mostly external code,
# adapted for the purpose of performing a *random* temperature change.
# source: https://www.askaswiss.com/2016/02/how-to-manipulate-color-temperature-opencv-python.html

# Creates a lookup table for a generic curve effect
def create_LUT_8UC1(x, y):
    spl = scipy.interpolate.UnivariateSpline(x, y)
    return spl(range(256))

def warm(img_bgr_in, incr_ch_lut, decr_ch_lut):
    # Step 1 : increase red, decrease blue
    c_b, c_g, c_r = cv2.split(img_bgr_in)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    img_bgr_warm = cv2.merge((c_b, c_g, c_r))
    # Step 2 : increase saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_warm, cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, incr_ch_lut).astype(np.uint8)
    img_bgr_warm = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)
    return img_bgr_warm

def cool(img_bgr_in, incr_ch_lut, decr_ch_lut):
    # Step 1 : increase blue, decrease red
    c_b, c_g, c_r = cv2.split(img_bgr_in)
    c_r = cv2.LUT(c_r, decr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, incr_ch_lut).astype(np.uint8)
    img_bgr_cold = cv2.merge((c_b, c_g, c_r))
    # Step 2 : decrease saturation
    c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_bgr_cold, cv2.COLOR_BGR2HSV))
    c_s = cv2.LUT(c_s, decr_ch_lut).astype(np.uint8)
    img_bgr_cold = cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2BGR)
    return img_bgr_cold

def random_temperature(img, rn_bounds=(0.0, 1.2)):       # , moneta_truccata=0.5):
    # Step 0 (ours) : random intensity
    x = np.array([0, 64, 128, 192, 256])
    rn = np.random.uniform(*rn_bounds)
    reference_incr_y = np.array([0, 70, 140, 210, 256])
    reference_decr_y = np.array([0, 30, 80, 120, 192])
    ref_incr_delta = reference_incr_y - x
    ref_decr_delta = reference_decr_y - x
    incr_delta = rn*ref_incr_delta
    decr_delta = rn*ref_decr_delta
    incr_y = x + incr_delta
    decr_y = x + decr_delta
    incr_ch_lut = create_LUT_8UC1(x, incr_y)
    decr_ch_lut = create_LUT_8UC1(x, decr_y)
    if np.random.rand() < 0.5:
        img = warm(img, incr_ch_lut, decr_ch_lut)
    else:
        img = cool(img, incr_ch_lut, decr_ch_lut)
    return img



# testing
if __name__=="__main__":
    img = cv2.imread(r"C:\Users\Giovanni\Universita\(M) Anno I\Neural Networks\DSC02107(2).jpg")
    warm_img = random_temperature(img, rn_bounds=(0.0, 1.2), moneta_truccata=999)
    cool_img = random_temperature(img, rn_bounds=(0.0, 1.2), moneta_truccata=-999)
    cv2.imwrite(r"C:\Users\Giovanni\Universita\(M) Anno I\Neural Networks\DSC02107(2)_warm.jpg", warm_img)
    cv2.imwrite(r"C:\Users\Giovanni\Universita\(M) Anno I\Neural Networks\DSC02107(2)_cool.jpg", cool_img)