import numpy as np
from sklearn.metrics import confusion_matrix

y_val_pred = np.load('Y_valid_predictions_7.npy')
y_val_ground = np.load('Y_valid.npy')

y_pred = np.load('Y_test_predictions_7.npy')
y_true = np.load('Y_test.npy')
# confusion_matrix = confusion_matrix(y_val_ground, y_val_pred)

ground = []
preds = []
for i in range(len(y_true)):
	ground.append(np.argmax(y_true[i]))
	preds.append(np.argmax(y_pred[i]))


# print(ground)
# print(preds)
confusion_matrix = confusion_matrix(ground, preds)

print(confusion_matrix)

def categorical_accuracy(prediction, ground_truth):
	return np.mean(np.equal(np.argmax(ground_truth, axis=-1), np.argmax(prediction, axis=-1)))

valid = categorical_accuracy(y_val_pred, y_val_ground)
print(valid)

test = categorical_accuracy(y_pred, y_true)
print(test)	


