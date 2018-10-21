import numpy as np
import keras.backend as K
from keras.utils import np_utils
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# lengths = np.load('lengths.npy')
def precision(matrix):
	avg_precise=0
	tp_sum = 0
	fp_sum = 0
	for i in range(4):
		tp = matrix[i][i]
		
		tp_sum+=tp 
		
		fp = np.sum(matrix[:, i]) 
		fp_sum+=fp
		if (fp!=0):

			avg_precise += tp/fp

	# print(tp_sum/4)
	# print(fp_sum/4)
	# print(tp_sum/fp_sum)
	return avg_precise/4



def recall(matrix):
	avg_recall=0
	tp_sum = 0
	fn_sum = 0
	for i in range(4):
		tp = matrix[i][i]
		tp_sum+=tp
		fn = np.sum(matrix[i, :]) 
		fn_sum+=fn
		avg_recall+=tp/fn
	# print(fn_sum/4)
	# print(tp_sum/fn_sum)
	return avg_recall/4



def f1_score(precision, recall):
	return (2*precision*recall)/(precision+recall)


def CrossEntropy(yHat, y):
    if y == 1:
      return -log(yHat)
    else:
      return -log(1 - yHat)


y_val_pred = np.load('Y_valid_predictions.npy')
y_val_ground = np.load('Y_valid_truth.npy')

length = len(y_val_pred)

# print(length)

confusion_matrix = confusion_matrix(y_val_ground, y_val_pred)

print(confusion_matrix)
# for i in range(length):
# 	if(y_val_pred[i]>=4):
# 		y_val_pred[i] -=4

# print(y_val_pred)
# np.save('Y_valid_predictions.npy', y_val_pred)

N_CLASSES = 4

# y_val_ground = np_utils.to_categorical(y_val_ground, N_CLASSES)
# y_val_pred = np_utils.to_categorical(y_val_pred, N_CLASSES)

# print(K.categorical_crossentropy(y_val_ground, y_val_pred))

precise = precision(confusion_matrix)
recall = recall(confusion_matrix)
f1 = f1_score(precise, recall)

print(precise, recall, f1)