

import numpy as np
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

model_path = os.environ['HOME'] + "/Triton/ensemble/models/xgb/1"

print("shape X " + str(X.shape))
print("shape Y " + str(Y.shape))

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy: {:.2f}".format(accuracy * 100.0))

### .save_config()
### print("config:")
### print(config)

###model.save_model(model_path + "/xgboost.json")

model.save_model(model_path + "/xgboost.model")
