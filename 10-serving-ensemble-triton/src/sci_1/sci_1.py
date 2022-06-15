

import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pickle

import subprocess


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

print("shape X " + str(X.shape))
print("shape Y " + str(Y.shape))

model_path = os.environ['HOME'] + "/Triton/ensemble/models/sci_1/1"

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X_train, y_train)
### print(model.predict([[0, 0, 0, 0]]))

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


pickle.dump(model, open(model_path + "/sci_1.pkl", 'wb'))

subprocess.run(["{}/Triton/ensemble/fil_backend/scripts/convert_sklearn".format(os.environ['HOME']), model_path + "/sci_1.pkl"])

