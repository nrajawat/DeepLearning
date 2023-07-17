#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[7]:


#[1]
# Use two Plots to show f1(x,y) and f2(x,y) values with respect to the two dimensional input x and y 
# (You can specify the range of the x and y values)

# Function definitions
def f1(x, y):
    return (x - 2) ** 2 + (y - 3) ** 2

def f2(x, y):
    return (1 - (y - 3)) ** 2 + 20 * ((x + 3) - (y - 3) ** 2) ** 2

# Gradient descent implementation
def gradient_descent(learning_rate, num_iterations):
    # Initialize variables
    x = np.random.uniform(-5, 5)  # Initial x value within the range
    y = np.random.uniform(-5, 5)  # Initial y value within the range
    
    # Lists to store the function values for plotting
    f1_values = []
    f2_values = []
    
    for _ in range(num_iterations):
        # Calculate gradients
        df1_dx = 2 * (x - 2)
        df1_dy = 2 * (y - 3)
        
        df2_dx = 40 * (x + 3 - (y - 3) ** 2)
        df2_dy = -2 * (1 - (y - 3)) - 80 * (x + 3 - (y - 3) ** 2) * (y - 3)
        
        # Update variables
        x -= learning_rate * df1_dx
        y -= learning_rate * df1_dy
        
        # Calculate function values and store for plotting
        f1_value = f1(x, y)
        f2_value = f2(x, y)
        
        f1_values.append(f1_value)
        f2_values.append(f2_value)
        
    return x, y, f1_values, f2_values

# Hyperparameters
learning_rate = 0.01
num_iterations = 100

# Run gradient descent
x_f1, y_f1, f1_values, _ = gradient_descent(learning_rate, num_iterations)
x_f2, y_f2, _, f2_values = gradient_descent(learning_rate, num_iterations)

# Plotting
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z_f1 = f1(X, Y)
Z_f2 = f2(X, Y)

fig = plt.figure(figsize=(12, 5))

# Plot f1(x, y)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z_f1, cmap='viridis', alpha=0.8)
ax1.scatter(x_f1, y_f1, f1_values[-1], color='red', label='Minimum')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f1(x, y)')
ax1.set_title('Function f1(x, y)')

# Plot f2(x, y)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z_f2, cmap='viridis', alpha=0.8)
ax2.scatter(x_f2, y_f2, f2_values[-1], color='red', label='Minimum')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f2(x, y)')
ax2.set_title('Function f2(x, y)')

plt.show()


# In[8]:


# [2]
# Starting from initial value: (x,y)=(0,0), use learning rate =0.5, report f1(x,y) and f2(x,y) values in T=100 iterations. 
#(your code can report f1(x,y) and f2(x,y) values as tables, or simple print out the values).Explain whether the
# gradient descent learning is effective finding the solutions for f1(x,y) and f2(x,y), why or why not?

# Gradient descent implementation (with value reporting)
def gradient_descent(learning_rate, num_iterations):
    # Initialize variables
    x = 0
    y = 0
    
    # Lists to store the function values for reporting
    f1_values = []
    f2_values = []
    
    for i in range(num_iterations):
        # Calculate gradients
        df1_dx = 2 * (x - 2)
        df1_dy = 2 * (y - 3)
        
        df2_dx = 40 * (x + 3 - (y - 3) ** 2)
        df2_dy = -2 * (1 - (y - 3)) - 80 * (x + 3 - (y - 3) ** 2) * (y - 3)
        
        # Update variables
        x -= learning_rate * df1_dx
        y -= learning_rate * df1_dy
        
        # Calculate function values and store for reporting
        f1_value = f1(x, y)
        f2_value = f2(x, y)
        
        f1_values.append(f1_value)
        f2_values.append(f2_value)
        
        print(f"Iteration {i+1}: f1(x, y) = {f1_value:.4f}, f2(x, y) = {f2_value:.4f}")
    
    return x, y, f1_values, f2_values

# Hyperparameters
learning_rate = 0.5
num_iterations = 100

# Run gradient descent and report values
x_f1, y_f1, f1_values, f2_values = gradient_descent(learning_rate, num_iterations)



# In[9]:


# [3]
# Following step 2, please change your code (e.g., using different learning rates, such as =0.01) to try to search 
# minimum for f1(x,y) and f2(x,y), respectively.Run algorithms for T=100 iterations Explain the motivation of your 
# changes, and the final minimum values.

# Gradient descent implementation (with value reporting)
def gradient_descent(learning_rate, num_iterations):
    # Initialize variables
    x = 0
    y = 0
    
    # Lists to store the function values for reporting
    f1_values = []
    f2_values = []
    
    for i in range(num_iterations):
        # Calculate gradients
        df1_dx = 2 * (x - 2)
        df1_dy = 2 * (y - 3)
        
        df2_dx = 40 * (x + 3 - (y - 3) ** 2)
        df2_dy = -2 * (1 - (y - 3)) - 80 * (x + 3 - (y - 3) ** 2) * (y - 3)
        
        # Update variables
        x -= learning_rate * df1_dx
        y -= learning_rate * df1_dy
        
        # Calculate function values and store for reporting
        f1_value = f1(x, y)
        f2_value = f2(x, y)
        
        f1_values.append(f1_value)
        f2_values.append(f2_value)
        
        print(f"Iteration {i+1}: f1(x, y) = {f1_value:.4f}, f2(x, y) = {f2_value:.4f}")
    
    return x, y, f1_values, f2_values

# Hyperparameters
learning_rate = 0.01
num_iterations = 100

# Run gradient descent and report values
x_f1, y_f1, f1_values, f2_values = gradient_descent(learning_rate, num_iterations)


# In[10]:


# [4]
# Explain why gradient descent learning can be used to help search solutions for f1(x,y) and f2(x,y), 
# and what are the impact of the learning rate in the gradient descent learning.

def gradient_descent_learning():
    """
    This function explains the usage of gradient descent learning to search for solutions
    and discusses the impact of the learning rate.
    """
    explanation = """Gradient descent learning is an iterative optimization algorithm that uses the gradient 
                     (partial derivatives) of a function to search for solutions. It is applicable to functions 
                     like f1(x, y) and f2(x, y) because it aims to minimize the value of the functions by 
                     descending along the function's surface. The basic idea is to update the input variables 
                     (x and y) iteratively in the opposite direction of the gradient until reaching a point 
                     where the gradient becomes close to zero, indicating a local minimum or global minimum."""
    
    impact_learning_rate = """The learning rate in gradient descent learning is a crucial parameter. It determines 
                              the step size taken in each iteration and has a significant impact on the optimization 
                              process. If the learning rate is too small, the algorithm may converge very slowly, 
                              requiring more iterations to reach the minimum. On the other hand, if the learning rate 
                              is too large, the algorithm may overshoot the minimum or even diverge, resulting in 
                              failed convergence. The learning rate acts as a trade-off between convergence speed 
                              and stability. A higher learning rate leads to faster convergence but risks overshooting 
                              and instability, while a lower learning rate ensures stability but slows down convergence. 
                              Choosing the appropriate learning rate involves finding a balance between these factors, 
                              often through experimentation and fine-tuning. Adaptive learning rate strategies can 
                              be employed to dynamically adjust the learning rate during optimization, achieving 
                              efficient and stable convergence."""
    
    print("Explanation:")
    print(explanation)
    
    print("\nImpact of the Learning Rate:")
    print(impact_learning_rate)

# Run the function
gradient_descent_learning()


# In[ ]:




