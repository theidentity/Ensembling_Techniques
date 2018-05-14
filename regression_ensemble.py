import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

# Order = [mlp,rbf,target]

def get_data():
	train = np.load('data/regression/train_ens.npy')
	X_train = train[:,:-1]
	y_train = train[:,-1]

	test = np.load('data/regression/test_ens.npy')
	X_test = test[:,:-1]
	y_test = test[:,-1]

	return X_train,y_train,X_test,y_test

def get_stats(y_true,y_pred):
	mse = mean_squared_error(y_true,y_pred)
	r2 = r2_score(y_true,y_pred)

	print 'MSE : ',mse
	print 'R2_score : ',r2

def mean_ensemble(X):
	output = np.mean(X,axis=1)
	return output

def get_stacking_model():
	model = MLPRegressor(hidden_layer_sizes=(20,20))
	X_train,y_train,_,_ = get_data()
	model.fit(X_train,y_train)
	return model

def stacking_ensemble(X):
	model = get_stacking_model()
	ens_stack_output = model.predict(X)
	return ens_stack_output

if __name__ == '__main__':
	X_train,y_train,X_test,y_test = get_data()
	
	print '------------- MLP------------- '
	get_stats(y_test,X_test[:,0])	

	print '------------- RBF------------- '
	get_stats(y_test,X_test[:,1])	

	print '------------- MEAN ENSEMBLE------------- '
	ens_mean_output = mean_ensemble(X_test)
	get_stats(y_test,ens_mean_output)	

	print '------------- STACKING ENSEMBLE------------- '
	ens_stack_output = stacking_ensemble(X_test)
	get_stats(y_test,ens_stack_output)	
