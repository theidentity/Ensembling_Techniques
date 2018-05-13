import numpy as np
import pandas as pd


capsnet = pd.read_csv('data/csv/capsnet.csv')
capsnet = np.array(capsnet,dtype=np.float64)
print capsnet.shape
print np.unique(capsnet)

cnn_custom = pd.read_csv('data/csv/custom_arch.csv')
cnn_custom = np.array(cnn_custom,dtype=np.float64)
print cnn_custom.shape
print np.unique(cnn_custom)

resnet = np.load('data/npy/ResNet50_pred.npy').astype(np.float64)
resnet = resnet[:-1]
print resnet.shape
print np.unique(resnet)

xception = np.load('data/npy/Xception_pred.npy').astype(np.float64)
xception = xception[:-1]
print xception.shape
print np.unique(xception)

pred_ensembles = np.array([resnet,xception,capsnet,cnn_custom],dtype=np.float64)
print pred_ensembles.shape
np.save('data/npy/pred_all.npy',pred_ensembles)

target = np.load('data/npy/ResNet50_true.npy').astype(np.float64)
target = np.argmax(target,axis=1)
target = target[:-1]
print target.shape
print np.unique(target)
np.save('data/npy/target.npy',target)
