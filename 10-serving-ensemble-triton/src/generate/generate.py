
import numpy as np
import os

features = 4
samples = 1000

data_path = os.environ['HOME'] + "/Triton/ensemble/data/"
X_data = data_path + 'X.data.npy'
Y_data = data_path + 'Y.data.npy'

if (os.path.exists(X_data)):
    print("Data already exists at {}".format(X_data))
else:
    print("Generating data at {}".format(data_path))
    X = np.random.rand(samples, features).astype('float32')
    Y = np.random.randint(2, size=samples)
    np.save(X_data, X)
    np.save(Y_data, Y)

