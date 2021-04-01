import tensorflow as tf
import tensorflow.keras as keras

MODEL_PATH = r"C:\Users\Giovanni\Universita\M_Anno_I\Neural Networks\models\Wiki no cascade\model_wiki_100epochs_nocascade.h5"

model = keras.models.load_model(MODEL_PATH, custom_objects={'tf': tf})

print(model.summary())