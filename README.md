
Here's a comprehensive guide on how to approach each of the questions you provided, along with sample Python code snippets for implementation. I'll also include potential viva questions and answers for your lab.

1. Plant Height and Sunlight Exposure
Sample Code:

python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from scipy.stats import pearsonr  

# Create synthetic data  
data = {  
    'Plant Name': ['Plant A', 'Plant B', 'Plant C', 'Plant D', 'Plant E'],  
    'Sunlight Exposure (hours)': [5, 6, 7, 8, 9],  
    'Plant Height (cm)': [15, 18, 21, 24, 30]  
}  
df = pd.DataFrame(data)  

# Scatter plot  
plt.figure(figsize=(10, 5))  
plt.scatter(df['Sunlight Exposure (hours)'], df['Plant Height (cm)'])  
plt.title('Sunlight Exposure vs Plant Height')  
plt.xlabel('Sunlight Exposure (hours)')  
plt.ylabel('Plant Height (cm)')  
plt.show()  

# Correlation coefficient  
correlation, _ = pearsonr(df['Sunlight Exposure (hours)'], df['Plant Height (cm)'])  
print(f'Correlation Coefficient: {correlation}')  

# Interpretation  
if correlation > 0:  
    print("Positive correlation")  
else:  
    print("Negative correlation")  

if abs(correlation) > 0.7:  
    print("Strong correlation")  
else:  
    print("Weak correlation")  
Questions and Answers for Viva
Q: How did you create the synthetic dataset?

A: I created a dictionary with plant names, sunlight exposure, and plant heights, then converted it to a DataFrame.
Q: What does the correlation coefficient indicate?

A: The correlation coefficient indicates the strength and direction of the linear relationship between two variables.
Q: How do you interpret a correlation coefficient of 0.85?

A: A correlation coefficient of 0.85 indicates a strong positive relationship between sunlight exposure and plant height.
2. Solar Panel Efficiency Study
Sample Code:

python
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
import statsmodels.api as sm  

# Create synthetic data  
np.random.seed(0)  
temperature = np.random.uniform(15, 35, 50)  
efficiency = 0.5 * temperature + np.random.normal(0, 2, 50)  

df_solar = pd.DataFrame({'Temperature (C)': temperature, 'Efficiency (%)': efficiency})  

# Train-test split  
X = df_solar[['Temperature (C)']]  
y = df_solar['Efficiency (%)']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Fit linear regression model  
model = LinearRegression()  
model.fit(X_train, y_train)  

# F-test  
X_with_const = sm.add_constant(X_train)  
model_sm = sm.OLS(y_train, X_with_const).fit()  
print(model_sm.summary())  

# T-test for the regression coefficient  
t_statistic = model_sm.tvalues[1]  
p_value = model_sm.pvalues[1]  
print(f'T-statistic: {t_statistic}, P-value: {p_value}')  
Questions and Answers for Viva
Q: What is the purpose of the F-test?

A: The F-test assesses whether the model significantly predicts the dependent variable.
Q: What does a low p-value indicate in the t-test?

A: A low p-value indicates that the regression coefficient is significantly different from zero, suggesting a significant relationship between temperature and efficiency.
3. Linear Regression Model for Students' Exam Scores
Steps:

Data Collection: Gather data on study hours and exam scores.
Data Preprocessing: Clean the data and handle missing values.
Model Building: Use LinearRegression to fit the model.
Assumption Checking: Check linearity, homoscedasticity, and normality of residuals.
Outlier Detection: Use box plots or Z-scores.
Model Evaluation: Calculate R², RMSE, and visualize predictions vs. actual scores.
Questions and Answers for Viva
Q: How do you check for linearity in your regression model?

A: I plot a scatter plot of the independent variable against the dependent variable to visually inspect for a linear relationship.
Q: What is R², and why is it important?

A: R² measures the proportion of variance in the dependent variable explained by the independent variable(s). It indicates the goodness of fit of the model.
4. Multiple Linear Regression Analysis
Sample Code:

python
# Create synthetic data  
data_retail = {  
    'Advertising Expenditure': [100, 150, 200, 250, 300],  
    'Store Location': [1, 2, 1, 2, 1],  
    'Competition': [5, 3, 4, 2, 1],  
    'Sales Revenue': [2000, 3000, 4000, 5000, 6000]  
}  
df_retail = pd.DataFrame(data_retail)  

# Fit multiple linear regression  
X = df_retail[['Advertising Expenditure', 'Store Location', 'Competition']]  
y = df_retail['Sales Revenue']  
X_with_const = sm.add_constant(X)  
model_retail = sm.OLS(y, X_with_const).fit()  
print(model_retail.summary())  

# Coefficient interpretation  
print(model_retail.params)  
Questions and Answers for Viva
Q: How do you interpret the coefficients in a multiple regression model?

A: Each coefficient indicates the expected change in the dependent variable for a one-unit change in the predictor, holding other variables constant.
Q: What does the F-test tell you in this context?

A: The F-test assesses the overall significance of the model, indicating whether at least one predictor is significantly related to the dependent variable.
5. k-Nearest Neighbors Classification
Sample Code:

python
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score  

# Load the Iris dataset  
iris = load_iris()  
X = iris.data  
y = iris.target  

# Train-test split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Fit kNN model  
knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X_train, y_train)  

# Predictions and accuracy  
y_pred = knn.predict(X_test)  
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')  

# Visualization  
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')  
plt.title('kNN Classification Results')  
plt.xlabel('Feature 1')  
plt.ylabel('Feature 2')  
plt.show()  
Questions and Answers for Viva
Q: What does k in kNN represent?

A: The parameter k represents the number of nearest neighbors to consider when making a prediction.
Q: How do you evaluate the performance of the kNN model?

A: I evaluate performance using accuracy, which measures the proportion of correct predictions.
6. K-means Clustering
Sample Code:

python
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt  

# Create synthetic customer data  
np.random.seed(0)  
data_customers = {  
    'Age': np.random.randint(18, 70, 100),  
    'Income': np.random.randint(20000, 100000, 100),  
    'Spending Score': np.random.randint(1, 100, 100)  
}  
df_customers = pd.DataFrame(data_customers)  

# Elbow Method to determine optimal clusters  
inertia = []  
for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i)  
    kmeans.fit(df_customers)  
    inertia.append(kmeans.inertia_)  

plt.figure(figsize=(10, 5))  
plt.plot(range(1, 11), inertia)  
plt.title('Elbow Method')  
plt.xlabel('Number of Clusters')  
plt.ylabel('Inertia')  
plt.show()  

# Fit K-means with optimal clusters  
optimal_k = 4  # Assuming 4 is optimal from the elbow method  
kmeans = KMeans(n_clusters=optimal_k)  
df_customers['Cluster'] = kmeans.fit_predict(df_customers)  

# Visualization of clusters  
plt.figure(figsize=(10, 5))  
plt.scatter(df_customers['Income'], df_customers['Spending Score'], c=df_customers['Cluster'], cmap='viridis')  
plt.title('K-means Clustering of Customers')  
plt.xlabel('Income')  
plt.ylabel('Spending Score')  
plt.show()  
Questions and Answers for Viva
Q: What is the purpose of the Elbow Method?

A: The Elbow Method helps to determine the optimal number of clusters by plotting the inertia against the number of clusters and finding the point where the decrease in inertia slows down.
Q: How do you interpret the clusters formed by K-means?

A: Each cluster represents a group of customers with similar characteristics, which can help in targeted marketing strategies.
Conclusion
This guide provides a structured approach to each question, complete with code snippets and potential viva questions and answers. Ensure you understand the concepts behind each step, as this will help you during your lab and any discussions with your instructors. Good luck!
