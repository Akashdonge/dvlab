import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("solar_efficiency_temp.csv")

# Display the first few rows of the dataset
print(df.head())

# Define predictor (temperature) and target (efficiency) variables
X = df[['temperature']]
y = df['efficiency']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Plot the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.title("Comparing Test Data with Predicted Data")
plt.xlabel("Temperature")
plt.ylabel("Solar Panel Efficiency")
plt.scatter(X_test, y_test, color="r", label="Actual Data")
plt.plot(X_test, y_pred, color="b", label="Predicted Data")
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error = {mse}\nr^2 = {r2}")

# Perform F-test and t-test using statsmodels
X_const = sm.add_constant(df[['temperature']])  # Add constant for intercept
Y = df['efficiency']

# Fit the model using statsmodels
sm_model = sm.OLS(Y, X_const).fit()

# Get t-statistic and p-value for the regression coefficient (temperature)
t_statistic = sm_model.tvalues['temperature']
p_value_t = sm_model.pvalues['temperature']

# Get F-statistic and its p-value
f_statistic = sm_model.fvalue
p_value_f = sm_model.f_pvalue

# Print results
print(f"F-statistic = {f_statistic}\nP-value (F-test) = {p_value_f}")
print(f"t-statistic = {t_statistic}\nP-value (t-test) = {p_value_t}")

if p_value_t < 0.05:
    print("The regression coefficient for temperature is statistically significant.")
else:
    print("The regression coefficient for temperature is NOT statistically significant.")

if p_value_f < 0.05:
    print("The temperature significantly predicts the efficiency of solar panels.")
else:
    print("The temperature does NOT significantly predict the efficiency of solar panels.")
