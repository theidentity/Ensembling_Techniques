import numpy as np
import pandas as pd


def npy_to_csv(inpath,outpath):
	pred = np.load(inpath)
	y_actual,y_pred = np.hsplit(pred,2)

	print y_actual
	print y_pred

	np.savetxt(outpath,y_pred,delimiter=',')
	np.savetxt('data/csv/actual.csv',y_actual,delimiter=',')

npy_to_csv('data/npy/ResNet50_.npy','data/csv/resnet50.csv')
npy_to_csv('data/npy/Xception_.npy','data/csv/xception.csv')

def combine_csv(root_path):

	in_csvs = ['custom_arch','resnet50','xception']
	in_csvs = [''.join([root_path,x,'.csv']) for x in in_csvs]
	csvs = [pd.read_csv(x) for x in in_csvs]
	
	target_csv = 'actual'
	target = pd.read_csv(''.join([root_path,target_csv,'.csv']))
	csvs.append(target)
	
	combined_csv = pd.concat([x for x in csvs], axis=1)
	print combined_csv.shape

	output_csv = 'combined'
	np.savetxt('data/csv/'+output_csv+'.csv',combined_csv,delimiter=',')


combine_csv('data/csv/')