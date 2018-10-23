import numpy as np
from sklearn.metrics import confusion_matrix

y_val_pred = np.load('Y_valid_predictions_2.npy')
y_val_ground = np.load('Y_valid.npy')

# confusion_matrix = confusion_matrix(y_val_ground, y_val_pred)

# print(confusion_matrix)


a = np.mean(np.equal(np.argmax(y_val_ground, axis=-1), np.argmax(y_val_pred, axis=-1)))
print(a)

