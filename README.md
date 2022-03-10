# Neural_Network_Charity_Analysis
Preprocessing, Compiling, Training, Evaluating, and Optimizing a Neural Network Model

## Overview of the Analysis:

#### Using Tensorflow, Keras, Pandas, Matplotlib, and Seaborn, an extensive assessment of potential charity funds was completed. A dataset containing more than 34,000 organizations was preprocessed and using a neural network, a model was compiled, trained and evaluated to create a target prediction accuracy  of >= 75% as to whether or not a charity would be successful or not if funded by a set donation amount. 

## Results:

### Data Preprocessing

What variable(s) are considered the target(s) for your model?

It is common practice to designate a machine learning model's target variable as 'y' when creating the variables for the training and testing dataset. For this analysis, the target variable was a binary column labeled `IS_SUCCESSFUL`. "0" denoting not successful after receiving a donation, and "1" denoting is successful after receiving a donation.

What variable(s) are considered to be the features for your model?

It is common practice to designate a machine learning model's features variable as 'X' when creating the variables for the training and testing dataset. For this analysis, the feature variable "X" was the entire dataframe's values, not inclusive of the target variable's column `IS_SUCCESSFUL`.

What variable(s) are neither targets nor features, and should be removed from the input data?

It is up to the analyst or engineer to determine what input features aren't suited to provide relevant information for the model. In this analysis, we looked for columns that lacked variability or uniqueness, and those columns were `STATUS`, `SPECIAL_CONSIDERATIONS` and `EIN`. Some students decided that the dataset's column `NAME` was suited to stay as an input variable as they found that the names of organizations actually held large weight to their models as seen by `.feature_importances_`. I decided to remove the `NAME` column as it suggested that there may be bias towards organizations that are well known.

### Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?

For the unoptimized model I selected the `number_input_features` to be the `len(X_train[0])` and created two hidden layers where the first hidden layer had neurons equal to the  `number_input_features` multiplied by 4, and the second hidden layer had neurons equal to the `number_input_features` multiplied by 3. I chose to modify rules of thumb (due to complexity of data) where the number of hidden neurons should be between the size of the input layer and the size of the output layer, or where hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer (https://medium.com/geekculture/introduction-to-neural-network-2f8b8221fbd3). Instead I chose a multiple of the input layer and then reduced it by 25% in the second hidden layer. The first hidden layer used the "relu" activation function and the 2nd hidden layer used the "sigmoid" activation function, as did the output layer. These activations are commonly used in binary classifications as is the "binary_crossentropy" loss function used when compiling the model.

Original, unoptimized model:

  ![notloss](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/fa3c443406f3e488577e9bd6c11194e7acd7cafa/README%20IMAGES/not_optimized_loss.png)<br>
  ![notaccuracy](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/fa3c443406f3e488577e9bd6c11194e7acd7cafa/README%20IMAGES/not_optimized_accuracy.png)<br>
  ![evaluateno](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/1174a22850f924446d36f12f327e8132a1815eac/README%20IMAGES/not_optimized_accuracy_results.png)<br><br>

Were you able to achieve the target model performance?

Looking at the results below, rounding to the hundredth place, I was able to achieve a predictive target accuracy of 75%.

![oploss](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/1174a22850f924446d36f12f327e8132a1815eac/README%20IMAGES/optimized_loss.png)<br>
![opaccuracy](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/1174a22850f924446d36f12f327e8132a1815eac/README%20IMAGES/optimized_accuracy.png)<br>
![opresults](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/1174a22850f924446d36f12f327e8132a1815eac/README%20IMAGES/optimized_accuracy_results.png)<br><br>

What steps did you take to try and increase model performance?

For the optimized model I:

- Benchmarked the dataset using a random forest classifier model to find the top 10 feature importances. <br>
![feautures](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/bd39d6d7608dce7665b7fc4c8a77168b2f4e6344/README%20IMAGES/top_10_features.png)<br>
- Once I saw that `ASK_AMT` was the most important feature I went and removed the outliers from that column and respective rows using pandas `.between`.<br>
![describe](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/eee50ac1475ca5e27609cb2e2666ca08eb0fbd4f/README%20IMAGES/describe.png)<br>
![outliers](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/eee50ac1475ca5e27609cb2e2666ca08eb0fbd4f/README%20IMAGES/outliers.png)<br>
![new_describe](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/fa3c443406f3e488577e9bd6c11194e7acd7cafa/README%20IMAGES/new_describe.png)<br>
- Added a 3rd hidden layer and more neurons to each layer, the first hidden layer is "relu" and all other layers are "sigmoid" activation functions.<br>
- Model weights were saved (every 5 epochs) as well as the .h5 file and different optimizers were chosen as well as epochs to optimize predictive accuracy.<br><br>

### Fun visuals

Confusion matrix from random forest classifier predictions.<br>
![cm](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/1174a22850f924446d36f12f327e8132a1815eac/README%20IMAGES/cm.png)<br><br>

Seaborn heatmap showing correlation among input features.<br>
![heatmap](https://github.com/derekhuggens/Neural_Network_Charity_Analysis/blob/1174a22850f924446d36f12f327e8132a1815eac/README%20IMAGES/sns_heatmap_corr.png)<br><br>

## Summary:

Using Tensorflow, Keras, Pandas, Matplotlib, and Seaborn, an extensive assessment of potential charity funds was completed. A dataset containing more than 34,000 organizations was preprocessed and using a neural network, a model was compiled, trained and evaluated to create a target prediction accuracy  of >= 75% as to whether or not a charity would be successful or not if funded by a set donation amount. As pointed out above, keeping the `NAME` column provided a way to bin less active funds and increase target predictive accuracies in other student's models but I did not want to include it for my own work. I decided to remove outliers from the input feature that gave the most weight to the model, `ASK_AMT`, which (while changing other hyperparamters), gave a model that achieved a target predictive accuracy of ~75.0%. 

Since this was a binary classification to predict whether a charity fund would be successul if their foundation ask amount was received, other binary classification models could be used. Within the `AlphabetSoupCharity_Optimization.ipynb`, one can find the random forest classifer being used to achieve a predictive accuracy of 74.5% when provided the same training set. Logistic regression and SVM could have been used as well as they both excel at binary classification problems.
