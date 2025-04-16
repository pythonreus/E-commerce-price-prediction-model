#!/usr/bin/env python
# coding: utf-8

# In[195]:


import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt


# In[196]:


#reading in my csv data and looking at the initial first 5 rows
data_frame = pd.read_csv("dataset.csv")
data_frame.head()


# In[197]:


#getting the length of my data frame
print(len(data_frame))


# In[198]:


#drop the email,address and avatar columns, not helpuf in the linear regression model
data_frame.drop(columns=["Email","Address","Avatar"], inplace= True) 
data_frame.head()


# In[199]:


#change all the column values to numeric, we are working with a linear regression model,numeric data is important
data_frame.apply(pd.to_numeric, errors="coerce") # if any value cannot be converted to numeric then just set it to NaN
#drop rows with missing values
data_frame.dropna(inplace=True) # setting inplace to be true means i am modifying the original data frame, changing the original dataset
#drop duplicates in the data frame
data_frame.drop_duplicates(inplace=True)
print(len(data_frame))


# In[200]:


#getting my features and my target column
features = data_frame.iloc[:, :4].values # getting all the rows and column 0 to 3
target_column = data_frame.iloc[:, 4].values # getting all the rows and column 4
print(target_column)


# In[201]:


# convert the money from us dollars to south african rands
def convert_dollar_to_rand(target_column):
    # one us dollar is equal to 18,83 rands
    Rand_Conversion = 18.83
    for i in range(len(target_column)):
        target_column[i] *= Rand_Conversion
    return target_column   
        
#target_column = convert_dollar_to_rand(target_column)   
#print(target_column)


# In[202]:


# splitting the dataset into training and testing data using scikit learn
# i will be using the 80-20 split ratio 
# random_state = 1 is here to ensure that the split is always the same,produces the same results each time
feature_training,feature_testing,target_training,target_testing = train_test_split(features,target_column,test_size=0.2, random_state=1)

#outputting the length of the training and the testing to see if the 80-20 ratio is correctly applied
print(target_training.shape[0])
print(target_testing.shape[0])


# In[203]:


# function to get the closed form solution
def closed_form_solution(feature_Matrix,target_Vector):
    theta = np.linalg.inv(feature_Matrix.T.dot(feature_Matrix)).dot(feature_Matrix.T).dot(target_Vector)
    return theta


# In[204]:


print(feature_training)


# In[205]:


#this is the filtering method by features
def filter_features(feature_Matrix,testing_Matrix,filter_list,feature_list):
    # get indices of selected features
    selected_indices = [feature_list.index(feature) for feature in filter_list]
    
    # get only the selected features for both training and testing
    selected_feature_Matrix = feature_Matrix[:, selected_indices]
    selected_testing_Matrix = testing_Matrix[:, selected_indices]
    
    # return the selected features
    return selected_feature_Matrix,selected_testing_Matrix
    


# In[206]:


# i have 4 features so this functions will return a list of all possible combinations i can have in this list
def get_all_combinations(feature_list):
    all_combinations = []  # This will store all combinations

    # Loop through different lengths of combinations (1 to len of list)
    for r in range(1, len(feature_list) + 1):
        # Generate combinations of length 'r' meaning if r is 2 it will create combinations of length 2
        current_combos = combinations(feature_list, r)
        
        # Convert each combo from tuple to list and add to final list
        for combo in current_combos:
            all_combinations.append(list(combo))
    
    return all_combinations


# In[207]:


def mean_squared_error(y_predict,y_value):
    divisor = len(y_predict)
    summation = 0
    for i in range(divisor):
        summation += math.pow((y_value[i] - y_predict[i]),2)
        
    return (summation/divisor)

def root_mean_squared_error(y_predict,y_value):
    return math.sqrt(mean_squared_error(y_predict,y_value))

def r_squared(y_predict,y_value):
    divisor = len(y_predict)
    summation = 0
    for i in range(divisor):
        summation += math.pow((y_value[i] - y_predict[i]),2)

    mean_value = sum(y_value) / divisor
    total = 0

    for i in range(divisor):
        total += math.pow((y_value[i] - mean_value), 2)

    result = 1 - (summation / total)
    return result
    
def mean_absolute_error(y_predict, y_value):
    divisor = len(y_predict)
    total_error = 0

    for i in range(divisor):
        total_error += abs(y_value[i] - y_predict[i])

    return total_error / divisor       


# In[208]:


# this is the function to plot the true values against the predicted values

def plot_predictions(y_value, y_predict, title="Predicted Values vs Actual Values"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_value, y_predict, color='royalblue', edgecolors='k', alpha=0.7)
    plt.plot([min(y_value), max(y_value)], [min(y_value), max(y_value)], color='red', linestyle='--')  # 1:1 line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[209]:


def get_residuals(y_value, y_predict):
    return [value - predict for value, predict in zip(y_value, y_predict)]


# In[210]:


# plotting the residuals
def plot_residuals(y_value, y_predict, title="Residual Plot"):
    residuals = get_residuals(y_value, y_predict)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_predict, residuals, color="darkorange", edgecolors='k', alpha=0.7)
    plt.axhline(y=0, color='blue', linestyle='--')  # horizontal line at 0
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[211]:


def plot_error_distribution(y_value, y_predict, title="Distribution of Prediction Errors"):
    residuals = get_residuals(y_value, y_predict)
    
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, color='teal', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# In[212]:


#this function shows the perfomance of the model with the selected features
class ModelResult:
    def __init__(self,filtered_list,y_predict,y_value,theta):
        self.filtered_list = filtered_list
        self.y_predict = y_predict
        self.y_value = y_value
        self.theta = theta
        self.MSE = None
        self.RMSE = None
        self.RSQUARED = None
        self.MAE = None

    def set_Metrics(self):
        self.set_MSE()
        self.set_RMSE()
        self.set_RSQUARED()
        self.set_MAE()
        
    def set_MSE(self):
        self.MSE = mean_squared_error(self.y_predict,self.y_value)

    def set_RMSE(self):
        self.RMSE = root_mean_squared_error(self.y_predict,self.y_value)

    def set_RSQUARED(self):
        self.RSQUARED = r_squared(self.y_predict,self.y_value)

    def set_MAE(self):
        self.MAE = mean_absolute_error(self.y_predict,self.y_value)
    
    def display_performance(self):
        print("")
        print("#############################################")
        print("")
        print(f'Filtered list: { self.filtered_list }')
        print(f'Theta values: { self.theta }')
        print(f'Mean Squared Error: { self.MSE }')
        print(f'Root Mean Squared Error: {self.RMSE}')
        print(f'R Squared: {self.RSQUARED}')
        print(f'Mean Absolute Error: {self.MAE}')
        print("")
        print("#############################################")
        print("")


# In[213]:


#this is the feature list
feature_list = ["Avg. Session Length","Time on App","Time on Website","Length of Membership"]
# get all the possible combinations from the list
possible_feature_combinations = get_all_combinations(feature_list)
print(possible_feature_combinations)


# In[214]:


#filter_list = possible_feature_combinations[10]
all_model_results = []

for filtered in possible_feature_combinations:
    #select which feature i want to train my model with
    filtered_features,filtered_testing = filter_features(feature_training,feature_testing,filtered,feature_list)
    # add a bias term { 1 } to our feature matrix inorder to be able to use the closed form solution so i am adding a new column of 1s
    #feature_training_bias = np.c_[np.ones((feature_training.shape[0],1)),feature_training]
    feature_training_bias = np.c_[np.ones((filtered_features.shape[0],1)),filtered_features]

    # get our theta values
    theta = closed_form_solution(feature_training_bias,target_training)

    # now i am testing the model, and i have to a column of 1s to my testing data
    feature_testing_bias = np.c_[np.ones((filtered_testing.shape[0],1)),filtered_testing]
    
    # and finally to test, i just get the dot product of the the feature_testing_bias with the theta vector i got
    tested = feature_testing_bias.dot(theta)
    #print(tested)
    all_model_results.append(ModelResult(filtered,tested,target_testing,theta))
    #plot_predictions(target_testing,tested)
    #display_performance(filtered,tested,target_testing,theta)


# In[215]:


# displaying all the preformances of all the models
print("")
print("\tModel Summary for closed form solution")
print("---------------------------------------------------------")
for model in all_model_results:
    model.set_Metrics()
    model.display_performance()


# In[216]:


# finding which model is the best
best_result = max(all_model_results, key=lambda model: model.RSQUARED)
print("\t Best Model")
best_result.display_performance()
plot_predictions(best_result.y_value,best_result.y_predict)
plot_residuals(best_result.y_value,best_result.y_predict)
plot_error_distribution(best_result.y_value,best_result.y_predict)


# In[217]:


# retrieve my top 3 best models
def best_Models():
    # Sort by R² descending
    top_results = sorted(all_model_results, key=lambda model: model.RSQUARED, reverse=True)[:3]
    
    # Display them
    for index, result in enumerate(top_results, 1):
        print(f"\n Top Model Number: {index}")
        result.display_performance()


# In[218]:


best_Models()


# In[ ]:


# now i am going to implement the gradient descent
def gradient_descent():
    pass

