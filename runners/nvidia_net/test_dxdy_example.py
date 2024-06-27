from matplotlib import image as mpimg
from config.config import record_saving_path
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd

"""
Epoch 1/10
300/300 [==============================] - 1540s 5s/step - loss: 0.0493 - val_loss: 0.0140
Epoch 2/10
300/300 [==============================] - 479s 2s/step - loss: 0.0146 - val_loss: 0.0100
Epoch 3/10
300/300 [==============================] - 221s 738ms/step - loss: 0.0124 - val_loss: 0.0081
Epoch 4/10
300/300 [==============================] - 177s 592ms/step - loss: 0.0107 - val_loss: 0.0073
Epoch 5/10
300/300 [==============================] - 134s 449ms/step - loss: 0.0094 - val_loss: 0.0058
Epoch 6/10
300/300 [==============================] - 122s 408ms/step - loss: 0.0082 - val_loss: 0.0048
Epoch 7/10
300/300 [==============================] - 112s 376ms/step - loss: 0.0072 - val_loss: 0.0045
Epoch 8/10
300/300 [==============================] - 112s 373ms/step - loss: 0.0065 - val_loss: 0.0040
Epoch 9/10
300/300 [==============================] - 110s 368ms/step - loss: 0.0058 - val_loss: 0.0037
Epoch 10/10
300/300 [==============================] - 107s 359ms/step - loss: 0.0052 - val_loss: 0.0029
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
Epoch 1/10
300/300 [==============================] - 826s 3s/step - loss: 0.0163 - val_loss: 0.0100
Epoch 2/10
300/300 [==============================] - 309s 1s/step - loss: 0.0114 - val_loss: 0.0082
Epoch 3/10
300/300 [==============================] - 224s 749ms/step - loss: 0.0092 - val_loss: 0.0055
Epoch 4/10
300/300 [==============================] - 192s 640ms/step - loss: 0.0079 - val_loss: 0.0048
Epoch 5/10
300/300 [==============================] - 159s 530ms/step - loss: 0.0068 - val_loss: 0.0040
Epoch 6/10
300/300 [==============================] - 146s 488ms/step - loss: 0.0060 - val_loss: 0.0034
Epoch 7/10
300/300 [==============================] - 143s 479ms/step - loss: 0.0055 - val_loss: 0.0031
Epoch 8/10
300/300 [==============================] - 142s 474ms/step - loss: 0.0051 - val_loss: 0.0029
Epoch 9/10
300/300 [==============================] - 141s 471ms/step - loss: 0.0047 - val_loss: 0.0026
Epoch 10/10
300/300 [==============================] - 140s 468ms/step - loss: 0.0043 - val_loss: 0.0023
"""

def img_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


columns = ['img', 'dx', 'dy']
data = pd.read_csv(record_saving_path + '/log.csv')

img_path = record_saving_path + '/img/' + data['img'].iloc[5]
img = mpimg.imread(img_path)
img = img_preprocess(img)
img = np.array([img], dtype=np.uint8)  # converting image into 4D array, as expected by the model
model = load_model(record_saving_path + '/model_dxdy.h5')
y_pred = model.predict(img, verbose=0)
print(y_pred[0])
