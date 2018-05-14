# Ensembling_Techniques

--------
## Ensembling for multiclass classification problems

### Methods Used 
#### Mean Ensembling
	* Takes the mean of the probability if the output classes
	* Output is the class with maximum probability

#### Majority Voting
	* Assigns as output whichever class has maximum number of votes among ensemble models
	* In case of a tie, takes the first group of tied classes

### Results
	
	Model | Accuracy
	--- | ---
	RESNET50 | 0.969882729211
	XCEPTION | 0.945895522388
	CAPSNET | 0.653251599147
	CNN_CUSTOM | 0.861407249467
	--- | ---
	MAJORITY VOTING | 0.967750533049
	MEAN ENSEMBLE | 0.982675906183
----

## Ensembling for regression problems
### Methods Used 
#### Mean Ensembling
	* Takes the mean of the predicted outputs by each model
	* Output is the class with maximum probability

#### Majority Voting
	* Assigns as output whichever class has maximum number of votes among ensemble models
	* In case of a tie, takes the first group of tied classes