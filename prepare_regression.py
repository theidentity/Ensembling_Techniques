import pandas as pd
import numpy as np
from glob import glob


# Order = [mlp,rbf,target]

def combine_train():
	mlp_train = np.load('data/regression/mlp_train.npy').reshape(-1,1)
	rbf_train = np.load('data/regression/rbf_train.npy').reshape(-1,1)
	train_X = np.hstack([mlp_train,rbf_train])
	return train_X

def combine_test():
	mlp_test = np.load('data/regression/mlp_test.npy').reshape(-1,1)
	rbf_test = np.load('data/regression/rbf_test.npy').reshape(-1,1)
	test_X = np.hstack([mlp_test,rbf_test])
	return test_X

def get_target_values():
	target_train = np.load('data/regression/trainY.npy')
	target_test = np.load('data/regression/testY.npy')
	return target_train,target_test

def save_data():
	train_X = combine_train()
	test_X = combine_test()
	target_train,target_test = get_target_values()

	train = np.hstack([train_X,target_train])
	test = np.hstack([test_X,target_test])

	np.save('data/regression/train_ens.npy',train)
	print train.shape
	np.save('data/regression/test_ens.npy',test)
	print test.shape

if __name__ == '__main__':
	save_data()