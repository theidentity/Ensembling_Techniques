# Ensembling_Techniques

## Ensembling for multiclass classification problems

### Methods Used 
1. Mean Ensembling
	* Takes the mean of the probability if the output classes
	* Output is the class with maximum probability

2. Majority Voting
	* Assigns as output whichever class has maximum number of votes among ensemble models
	* In case of a tie, takes the first group of tied classes

### Results
	
 | Model | Accuracy | 
 | --- | --- |
 | RESNET50 | 0.969882729211 |
 | XCEPTION | 0.945895522388 |
 | CAPSNET | 0.653251599147 |
 | CNN_CUSTOM | 0.861407249467 |
 | MAJORITY VOTING | 0.967750533049 |
 | MEAN ENSEMBLE | 0.982675906183 |
----

## Ensembling for regression problems
### Methods Used 
1. Mean Ensembling
	* Takes the mean of the predicted outputs by each model

2. Stacking
	* Takes the output of the ensembles and passed it through a MLP Regressor 
	* Output is the output provided by the regressor on passing the test data

### Results

 | Model | MSE | R2_Score |
 | --- | --- | --- |
 | MLP | 0.0103210967873 | 0.808796324494 |
 | RBF | 0.0106966198534 | 0.801839564767 |
 | MEAN ENSEMBLE | 0.00845186510476 | 0.843424811702 |
 | STACKING ENSEMBLE | 0.00729725486593 | 0.864814565717 |
