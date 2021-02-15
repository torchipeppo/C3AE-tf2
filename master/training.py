import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import model as model_module
import tensorflow.keras as keras
import os
import path_constants
from datetime import datetime
import pickle
import math
import cv2
from dataprocessing import COLS

def get_image_crops(row_with_index, seed, augment, nn_input_shape=(64,64)):
    index, row = row_with_index[0], row_with_index[1]
    image = np.frombuffer(row.image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if augment:
        pass   # TODO random erasing qui

    crops = []
    for box in np.loads(row.crop_boxes, encoding="bytes"):
        h0, w0 = np.int64(box[0])   # conversione np da float a int
        h1, w1 = np.int64(box[1])
        crops.append(cv2.resize(image[h0:h1,w0:w1], nn_input_shape))
    
    if augment:
        pass   # TODO altra augmentation qui

    return crops

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

def datagen(dataframe, batch_size, seed, categories, interval, augment):
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
                get_image_crops(rwi, seed, augment) for rwi in row_batch.iterrows()
            ])
            age_array = row_batch.age.to_numpy()
            values = [
                age_array,
                np.array([age2twopoint(a, categories, interval) for a in age_array])
            ]
            yield [image_batch[:,0], image_batch[:,1], image_batch[:,2]], values
            start += batch_size

def do_train(
    dataset, 
    train_split=0.888, 
    batch_size=50, 
    lr=0.002,
    loss_weight_factor=10,
    seed=14383421,
    bins=10,
    epochs=1   # 250
):
    categories = bins+2
    interval = int(math.ceil(100.0/bins))

    trainset, validset = train_test_split(
        dataset, train_size=train_split
    )
    train_gen = datagen(trainset, batch_size, seed, categories, interval, True)
    valid_gen = datagen(validset, batch_size, seed, categories, interval, False)

    model = model_module.full_context_model(bottleneck_dim=categories)

    opti = keras.optimizers.Adam(lr=lr)

    model.compile(
        optimizer=opti,
        loss=["mae", "kullback_leibler_divergence"],
        metrics={"age":"mae"},
        loss_weights=[1,loss_weight_factor]
    )

    history = model.fit_generator(
        train_gen,
        steps_per_epoch = len(trainset)/batch_size,
        epochs=1,   # 250
        validation_data=valid_gen,
        validation_steps = len(validset)/batch_size*3
    )

    now = datetime.now()
    model_fname = "model_{}_{}_{}_{}_{}_{}.h5".format(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second
    )
    model_path = os.path.join(path_constants.MODEL_HISTORY_SAVE, model_fname)
    model.save(model_path)
    history_fname = "history_{}_{}_{}_{}_{}_{}.hist".format(
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

if __name__=="__main__":
    files = os.listdir(path_constants.FGNET_PICKLE)
    frames=[]
    for fname in files:
        path = os.path.join(path_constants.FGNET_PICKLE, fname)
        frame = pd.read_pickle(path)
        frames.append(frame)
    dataset = pd.concat(list(frames), ignore_index=True)
    # filtriamo le età che non hanno senso
    dataset = dataset[(dataset.age>=0) & (dataset.age<=120)]
    dataset = dataset.dropna()

    do_train(dataset)
    