import pandas as pd
import matplotlib.pyplot as plt

# Create the dataset
data = {
    'plant_name': ["Tomato", "Lemon", "Capsicum", "Mulberry", "Persimmon", "Passion Fruit"],
    'sunlight_exposure': [20, 56, 18, 98, 34, 95],
    'plant_height': [67, 89, 12, 101, 45, 121]
}

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print("Dataset:")
print(df.head())

# Visualize the relationship between sunlight exposure and plant height using a scatterplot
plt.scatter(df['sunlight_exposure'], df['plant_height'], color="r")
plt.title("Relationship between Sunlight Exposure and Plant Height")
plt.xlabel("Sunlight Exposure (hours)")
plt.ylabel("Height (cm)")
plt.show()

# Calculate the correlation coefficient between sunlight exposure and plant height
reduced_df = df[['sunlight_exposure', 'plant_height']]
correlation_matrix = reduced_df.corr()
corr_coeff = correlation_matrix.loc['sunlight_exposure', 'plant_height']

# Print the correlation matrix and the correlation coefficient
print("\nCorrelation Matrix:")
print(correlation_matrix)
print(f"\nCorrelation Coefficient: {corr_coeff}")

# Determine the sign and strength of the correlation
if corr_coeff < 0:
    sign = "negative"
elif corr_coeff > 0:
    sign = "positive"
else:
    sign = "neither"

strength = "strong" if abs(corr_coeff) > 0.5 else "weak"

# Print correlation analysis
print(f"The correlation coefficient is {sign}.")
print(f"The correlation is {strength}.")

# Determine if there is a relationship between sunlight exposure and plant height
if abs(corr_coeff) > 0:
    print(f"Yes, there is a {strength} {sign} linear relationship between Sunlight Exposure and Plant Height.")
else:
    print("There is no relationship between Sunlight Exposure and Plant Height.")

# Based on the correlation coefficient, assess the significance of the association
if strength == "strong":
    print("Yes, we can conclude that there is significant association between Sunlight Exposure and Plant Height.")
elif strength == "weak":
    print("The association between Sunlight Exposure and Plant Height is not significant.")
elif sign == "neither":
    print("There is no association between Sunlight Exposure and Plant Height.")
