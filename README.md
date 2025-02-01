# Plant Growth and Solar Panel Efficiency Analysis  

This repository contains Python scripts and Jupyter notebooks that explore various data analysis and machine learning techniques. The analysis covers the relationship between plant growth and sunlight exposure, solar panel efficiency based on temperature, student performance based on study hours, retail sales influenced by advertising, and classification of flower types using the k-Nearest Neighbors algorithm.  

## Table of Contents  
1. [Plant Growth Analysis](#plant-growth-analysis)  
2. [Solar Panel Efficiency Study](#solar-panel-efficiency-study)  
3. [Student Performance Prediction](#student-performance-prediction)  
4. [Retail Sales Analysis](#retail-sales-analysis)  
5. [Flower Classification](#flower-classification)  
6. [Customer Segmentation](#customer-segmentation)  

## Plant Growth Analysis  

### Objective  
Investigate the relationship between sunlight exposure and the height of plants.  

### Tasks  
- **Relationship Analysis**: Determine if there is a relationship between sunlight exposure (in hours) and plant height (in cm).  
- **Visualization**: Create a scatterplot to visualize the relationship.  
- **Correlation Coefficient**: Calculate the correlation coefficient to assess the strength and direction of the relationship.  
- **Significance Assessment**: Conclude whether there is a significant association between sunlight exposure and plant growth.  

 Code  
```python  
 Sample Code for Plant Growth Analysis  
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np  

# Create a synthetic dataset  
data = {  
    'plant_name': ['Plant A', 'Plant B', 'Plant C', 'Plant D', 'Plant E'],  
    'sunlight_exposure': [5, 8, 12, 6, 10],  # hours  
    'plant_height': [15, 20, 30, 18, 25]  # cm  
}  
df = pd.DataFrame(data)  

# Scatterplot  
plt.scatter(df['sunlight_exposure'], df['plant_height'])  
plt.xlabel('Sunlight Exposure (hours)')  
plt.ylabel('Plant Height (cm)')  
plt.title('Sunlight Exposure vs. Plant Height')  
plt.show()  

# Correlation Coefficient  
correlation = df['sunlight_exposure'].corr(df['plant_height'])  
print(f'Correlation Coefficient: {correlation}')  
Solar Panel Efficiency Study
Objective
Investigate the relationship between temperature and the efficiency of solar panels.

Tasks
Linear Regression Model: Develop a model to predict solar panel efficiency based on temperature.
F-test: Perform an F-test to determine the significance of temperature as a predictor.
t-test: Conduct a t-test to assess the significance of the regression coefficient for temperature.
Code
python
# Sample Code for Solar Panel Efficiency Study  
import pandas as pd  
import statsmodels.api as sm  

# Create synthetic data  
data = {  
    'temperature': np.random.uniform(15, 35, 50),  # in Celsius  
    'efficiency': np.random.uniform(70, 100, 50)  # in percentage  
}  
df = pd.DataFrame(data)  

# Simple Linear Regression  
X = df['temperature']  
y = df['efficiency']  
X = sm.add_constant(X)  # Adds an intercept term  
model = sm.OLS(y, X).fit()  

# F-test and t-test results  
print(model.summary())  
Student Performance Prediction
Objective
Build a linear regression model to predict exam scores based on study hours.

Tasks
Model Building: Create a linear regression model.
Diagnostics: Check assumptions, identify outliers, and handle influential points.
Model Evaluation: Evaluate performance and discuss insights.
Steps
Load the dataset.
Explore the data and check for assumptions (linearity, normality).
Identify outliers using residual plots.
Build the regression model and evaluate its performance.
Retail Sales Analysis
Objective
Analyze how advertising expenditure, store location, and competition affect sales revenue.

Tasks
Multiple Linear Regression: Implement a regression model using synthetic data.
Coefficient Interpretation: Interpret the coefficients of the model.
F-test and t-tests: Assess model significance and individual coefficient significance.
Code
python
# Sample Code for Retail Sales Analysis  
import pandas as pd  
import statsmodels.api as sm  

# Create synthetic data  
data = {  
    'advertising_expenditure': np.random.uniform(1000, 5000, 100),  
    'store_location': np.random.randint(1, 5, 100),  
    'competition': np.random.uniform(0, 1, 100),  
    'sales_revenue': np.random.uniform(5000, 20000, 100)  
}  
df = pd.DataFrame(data)  

# Multiple Linear Regression  
X = df[['advertising_expenditure', 'store_location', 'competition']]  
y = df['sales_revenue']  
X = sm.add_constant(X)  
model = sm.OLS(y, X).fit()  

# Model summary  
print(model.summary())  
Flower Classification
Objective
Perform classification using the k-Nearest Neighbors (kNN) algorithm on the Iris dataset.

Tasks
Model Training: Train a k-NN classifier.
Performance Evaluation: Calculate accuracy and visualize results.
Code
python
# Sample Code for Flower Classification  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  
import matplotlib.pyplot as plt  

# Load Iris dataset  
iris = load_iris()  
df = pd.DataFrame(iris.data, columns=iris.feature_names)  
df['target'] = iris.target  

# Train-test split  
X = df.drop('target', axis=1)  
y = df['target']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  

# k-NN Classifier  
knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X_train, y_train)  
accuracy = knn.score(X_test, y_test)  
print(f'Accuracy: {accuracy * 100:.2f}%')  
Customer Segmentation
Objective
Perform K-means clustering on customer data.

Tasks
Data Preparation: Use customer data (e.g., Age, Income).
Clustering: Apply K-means clustering and visualize results.
Elbow Method: Determine the optimal number of clusters.
Code
python
# Sample Code for Customer Segmentation  
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt  

# Create synthetic customer data  
data = {  
    'Age': np.random.randint(18, 70, 100),  
    'Income': np.random.randint(20000, 120000, 100)  
}  
df = pd.DataFrame(data)  

# Elbow Method  
sse = []  
k_range = range(1, 11)  
for k in k_range:  
    kmeans = KMeans(n_clusters=k)  
    kmeans.fit(df)  
    sse.append(kmeans.inertia_)  

plt.plot(k_range, sse)  
plt.xlabel('Number of Clusters')  
plt.ylabel('SSE')  
plt.title('Elbow Method')  
plt.show()  

# Applying K-means with optimal clusters  
optimal_k = 3  
kmeans = KMeans(n_clusters=optimal_k)  
df['Cluster'] = kmeans.fit_predict(df)  

# Visualization  
plt.scatter(df['Age'], df['Income'], c=df['Cluster'])  
plt.xlabel('Age')  
plt.ylabel('Income')  
plt.title('Customer Segmentation')  
plt.show()  
