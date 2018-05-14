import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


# order of models : [resnet,xception,capsnet,cnn_custom]


def normalize_array(arr,low=0,high=1):
	arr = ((high-low)*(arr-np.min(arr)))/(np.max(arr)-np.min(arr))
	return arr

def prep_ensembles(ensembles):
	models = []
	for x in ensembles:
		models.append(normalize_array(x,0,1))
	models = np.array(models,dtype=np.float64)
	return models

def get_stats(y_pred,y_actual):
	cm = confusion_matrix(y_actual,y_pred)
	report = classification_report(y_actual,y_pred)
	accuracy = accuracy_score(y_actual,y_pred)

	print cm
	print report
	print 'Accuracy :',accuracy

def get_model_stats(models,target):
	names = ['resnet50','xception','capsnet','cnn_custom']
	for i,model in enumerate(models):
		print names[i].upper()
		y_pred = np.argmax(model,axis=1)
		y_true = target
		get_stats(y_pred,y_true)

def ensemble_mean(ensembles):
	pred = np.mean(ensembles,axis=0)
	output = np.argmax(pred,axis=1)
	return output

def ensemble_majority_voting(ensembles):
	ensembles = np.array([np.argmax(x,axis=1) for x in ensembles])
	voting = np.array([np.argmax(np.bincount(ensembles[:,x])) for x in range(ensembles.shape[1])])
	return voting



if __name__ == '__main__':
	
	pred_ensembles = np.load('data/npy/pred_all.npy').astype(np.float64)
	pred_ensembles = prep_ensembles(pred_ensembles)
	target = np.load('data/npy/target.npy')


	get_model_stats(pred_ensembles,target)

	print 'Mean ensembles'
	output = ensemble_mean(pred_ensembles)
	get_stats(output,target)

	print 'Majority Voting'
	output = ensemble_majority_voting(pred_ensembles)
	get_stats(output,target)


	# RESNET50
	# Accuracy : 0.969882729211
	# XCEPTION
	# Accuracy : 0.945895522388
	# CAPSNET
	# Accuracy : 0.653251599147
	# CNN_CUSTOM
	# Accuracy : 0.861407249467
	# Mean ensembles
	# Accuracy : 0.982675906183
	# Majority Voting
	# Accuracy : 0.967750533049