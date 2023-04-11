# deep_learning_challenge

# ANALYSIS OVERVIEW

The purpose of the analysis is to create a deep learning neural network model to predict whether or not the charities were successful when applying for the grant. The information that we used to train this model was from the charities data csv file as shown below.
![Screenshot 2023-04-11 214550](https://user-images.githubusercontent.com/115653868/231287356-17c1fc46-4102-410e-8ca5-3fde6191a11c.png)


# Data Preprocessing
The variabe the tos the target for this model is the IS_SUCCESSFUL variable as that is the variable that we are looking to predict using our model. The Faetures of the model are the remaining columns in the dataframe execpt for the IS_SUCCESSFUL variable as that is the target. I have also removed the EIN column as this is neither a target or a feature of this dataset.

# Split our preprocessed data into our features and target arrays
y = dummy_df['IS_SUCCESSFUL'].values
X = dummy_df.drop('IS_SUCCESSFUL', axis=1)


# Split the preprocessed data into a training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)


# Compiling, Training, and Evaluating the Model 

After optimising this model I have used 5 layers in the Neural network which consit of a total of 430 nodes. The model that I have used consists of the relu activation function before finally going to a sigmoid activation function as well are doing classification. I have used the relu function as this gave me the best results when comparing to the other activation functions and also because it does not activate all the neurons at the same time.

![Screenshot 2023-04-11 221648](https://user-images.githubusercontent.com/115653868/231290237-18004b42-ec9f-4f96-af95-d46c3499d521.png)


My model has achieved over the target model performance achieving nearly 75% accuracy. In order to increase my model performance, I have reduce the amount of columns that were dropped, changed the classifaction bins to include more information, increased the amount of layers and nodes, change the activation functions and also reduced the number of epochs.

# Summary
The overall result of my model were very positive a achieving well over the target performance. I would also recommend a logistic regression model to solve this classifcation problem. Logistic regression estimates the probability of an event occurrin, based on a given dataset of independent variables therefore would be suitable for this classification problem

