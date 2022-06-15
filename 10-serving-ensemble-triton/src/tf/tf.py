

import numpy as np
import os

import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc

from sklearn.model_selection import train_test_split


class Round(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Round, self).__init__()
        self.num_outputs = num_outputs

    def call(self, inputs):
        #print("inputs:" + str(inputs))
        #outputs = inputs.__floordiv__(1.0)
        outputs = inputs
        return outputs



seed = 7
features = 4
samples = 1000

data_path = os.environ['HOME'] + "/Triton/ensemble/data/"
X_data = data_path + 'X.data.npy'
Y_data = data_path + 'Y.data.npy'

if (not os.path.exists(X_data)):
    print("Please run src/generate.py to create dummy data for modes")
else:
   X = np.load(X_data)
   Y = np.load(Y_data)

print("s)hape X " + str(X.shape))
print("shape Y " + str(Y.shape))

model_path = os.environ['HOME'] + "/Triton/ensemble/models/tf/1"

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


model =  tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(4, input_dim=4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.add(Round(1))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.fit(X_train, y_train)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# creates /models/tf/model.tf/saved_model.pb
tf.saved_model.save(model, model_path + "/model.savedmodel")






