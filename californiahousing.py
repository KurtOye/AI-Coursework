from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the California housing dataset
california_housing = fetch_california_housing()

# Features x and target y
x = california_housing.data
y = california_housing.target

# Convert to dataframe for easy handling
df = pd.DataFrame(x, columns=california_housing.feature_names)
df['MedHouseVal'] = y

# Prepare the data for regression
X = df[['MedInc']].values
y = df['MedHouseVal'].values


# Add bias (intercept) term to X by adding a column of ones
X = np.c_[np.ones(X.shape[0]), X]

# Use train_test_split to split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Batch Gradient Descent Function
def batch_gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):  
        # Calculate the predictions
        predictions = X.dot(theta)
        
        # Calculate the gradient
        gradient = (1 / m) * X.T.dot(predictions - y)
        
        # Update theta
        theta = theta - learning_rate * gradient
        
    return theta

# Initialise theta (weights) to zero
theta = np.zeros(X_train.shape[1])

# Hyperparameters for BGD
learning_rate = 0.01
iterations = 1000

# Run the Batch Gradient Descent
theta_bgd = batch_gradient_descent(X_train, y_train, theta, learning_rate, iterations)

# Stochastic Gradient Descent Function
def stochastic_gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):  
        # Randomly pick one example
        random_index = np.random.randint(m)
        X_i = X[random_index:random_index+1]
        y_i = y[random_index:random_index+1]
        
        # Make a prediction for this example
        prediction = X_i.dot(theta)
        
        # Compute the gradient for this example
        gradient = X_i.T.dot(prediction - y_i)
        
        # Update the weights
        theta = theta - learning_rate * gradient
    
    return theta

# Initialise theta (weights) to zero for SGD
theta_sgd = np.zeros(X_train.shape[1])

# Run the Stochastic Gradient Descent
theta_sgd = stochastic_gradient_descent(X_train, y_train, theta_sgd, learning_rate, iterations)

# Make predictions using both models
y_pred_bgd = X_test.dot(theta_bgd)
y_pred_sgd = X_test.dot(theta_sgd)

# Plot the regression lines for both BGD and SGD along with the test data
plt.figure(figsize=(10, 6))

# Plot the test data
plt.scatter(X_test[:, 1], y_test, alpha=0.5, label='Test data')

# Plot the regression lines
plt.plot(X_test[:, 1], y_pred_bgd, label='BGD Regression Line', color='red', linewidth=2)
plt.plot(X_test[:, 1], y_pred_sgd, label='SGD Regression Line', color='blue', linewidth=2)


plt.title('Median Income vs Median House Value')
plt.xlabel('Median Income ($10,000s)')  
plt.ylabel('Median House Value ($100,000s)') 

# Set the x and y axis limits based on the data range
plt.xlim(0, X_test[:, 1].max() + 1) 
plt.ylim(0, y_test.max() + 1)  
plt.grid(True)
plt.legend()
plt.show()

# Summary statistics for 'MedInc' and 'MedHouseVal'
print(f"Mean of MedInc: {df['MedInc'].mean()}")
print(f"Median of MedInc: {df['MedInc'].median()}")
print(f"Standard Deviation of MedInc: {df['MedInc'].std()}")
print(f"Mean of MedHouseVal: {df['MedHouseVal'].mean()}")
print(f"Median of MedHouseVal: {df['MedHouseVal'].median()}")
print(f"Standard Deviation of MedHouseVal: {df['MedHouseVal'].std()}")


# Prediction of Median House Value at Median Income of $80,000
y_pred_bgd_test = X_test.dot(theta_bgd)
y_pred_sgd_test = X_test.dot(theta_sgd)
new_input = np.array([1, 8.0])  

# Prediction using both BGD and SGD Models
pred_bgd_new_input = new_input.dot(theta_bgd)
pred_sgd_new_input = new_input.dot(theta_sgd)  

# Printed Prediction
print(f"Predicted House Value (BGD) for MedInc = $80,000: ${pred_bgd_new_input * 100000:.2f}")
print(f"Predicted House Value (SGD) for MedInc = $80,000: ${pred_sgd_new_input * 100000:.2f}")

# Median income alone is not a good determiner of Median House Value because there are many other variables to consider.
# Such as coastal vs inland, coastal houses have higher demand because of their scenice views, access to beaches are other amenities.
# this would influence the house price regardless of median income. This is also for Urban houses that have better access to infrastructure and job opportunities etc.
# Besdies this you have the age of the property,  size of the property and proximity to major cities.
# To make this model more accurate, we can include the house age, and average rooms per household which have a large influence on the cost of a house.
# Then we can talk about the location of the house, the population in the area, and proximity to amenities.
# Including all of these would make the model much more accurate it it's predictions since it would reflect the factors that influence the house price in reality.