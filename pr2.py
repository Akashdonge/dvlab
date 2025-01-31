import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.linear_model import LinearRegression  
from sklearn.model_selection import train_test_split  
import statsmodels.api as sm  

# Load dataset  
df = pd.read_csv("solar_efficiency_temp.csv")  

# Display the first few rows of the dataset to understand its structure  
print(df.head())  

# Define predictor (temperature) and target (efficiency) variables  
X = df[['temperature']]  # Predictor variable(s)  
y = df['efficiency']     # Target variable  

# Split data into training and testing sets (75% train, 25% test)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)  

# Create and fit the Linear Regression model on training data  
model = LinearRegression()  
model.fit(X_train, y_train)  

# Predict on the test set using the trained model  
y_pred = model.predict(X_test)  

# Plot the actual vs predicted values for visualization  
plt.title("Comparing Test Data with Predicted Data")  # Set the plot title  
plt.xlabel("Temperature")  # Label for the x-axis  
plt.ylabel("Solar Panel Efficiency")  # Label for the y-axis  
plt.scatter(X_test, y_test, color="r", label="Actual Data")  # Scatter plot of actual values  
plt.plot(X_test, y_pred, color="b", label="Predicted Data")  # Line plot for predicted values  
plt.legend()  # Show legend  
plt.show()  # Display the plot  

# Evaluate the model using Mean Squared Error and R-squared metrics  
mse = mean_squared_error(y_test, y_pred)  # Calculate MSE  
r2 = r2_score(y_test, y_pred)  # Calculate R-squared  
print(f"Mean Squared Error = {mse}\nr^2 = {r2}")  # Print evaluation metrics  

# Perform F-test and t-test using statsmodels  
X_const = sm.add_constant(df[['temperature']])  # Add constant term for intercept in the regression model  
Y = df['efficiency']  # Target variable  

# Fit the Ordinary Least Squares (OLS) regression model  
model = sm.OLS(Y, X_const).fit()  

# Get t-statistic and p-value for the regression coefficient (temperature)  
t_statistic = model.tvalues['temperature']  # t-statistic for temperature  
p_value_t = model.pvalues['temperature']  # p-value for temperature  

# Get F-statistic and its p-value from the model  
f_statistic = model.fvalue  # F-statistic  
p_value_f = model.f_pvalue  # p-value for F-test  

# Print the statistical test results  
print(f"F-statistic = {f_statistic}\nt-statistic = {t_statistic}")  

# Check significance for the t-test  
if p_value_t < 0.05:  
    print("The regression coefficient for temperature is statistically significant.")  
else:  
    print("The regression coefficient for temperature is NOT statistically significant.")  

# Check significance for the F-test  
if p_value_f < 0.05:  
    print("The temperature significantly predicts the efficiency of solar panels.")  
else:  
    print("The temperature does NOT significantly predict the efficiency of solar panels.")
