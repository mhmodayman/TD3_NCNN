from matplotlib import image as mpimg
from config.config import record_saving_path
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd

"""
Epoch 1/10
400/400 [==============================] - 371s 920ms/step - loss: 0.0365 - val_loss: 7.5599e-04
Epoch 2/10
400/400 [==============================] - 147s 369ms/step - loss: 9.7219e-04 - val_loss: 5.4890e-04
Epoch 3/10
400/400 [==============================] - 142s 357ms/step - loss: 8.4522e-04 - val_loss: 5.5825e-04
Epoch 4/10
400/400 [==============================] - 145s 364ms/step - loss: 7.5753e-04 - val_loss: 4.7925e-04
Epoch 5/10
400/400 [==============================] - 141s 354ms/step - loss: 6.9619e-04 - val_loss: 3.9050e-04
Epoch 6/10
400/400 [==============================] - 140s 350ms/step - loss: 6.9493e-04 - val_loss: 3.9185e-04
Epoch 7/10
400/400 [==============================] - 141s 353ms/step - loss: 6.6788e-04 - val_loss: 4.3415e-04
Epoch 8/10
400/400 [==============================] - 142s 355ms/step - loss: 6.3242e-04 - val_loss: 3.6682e-04
Epoch 9/10
400/400 [==============================] - 143s 357ms/step - loss: 6.1602e-04 - val_loss: 3.5269e-04
Epoch 10/10
400/400 [==============================] - 143s 359ms/step - loss: 5.7690e-04 - val_loss: 2.8478e-04
"""

def img_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


columns = ['img', 'steer']
data = pd.read_csv(record_saving_path + '/log.csv')

img_path = record_saving_path + '/img/' + data['img'].iloc[5]
img = mpimg.imread(img_path)
img = img_preprocess(img)
img = np.array([img], dtype=np.uint8)  # converting image into 4D array, as expected by the model
model = load_model(record_saving_path + '/model_steer.h5')
y_pred = model.predict(img, verbose=0)
print(y_pred[0][0])
