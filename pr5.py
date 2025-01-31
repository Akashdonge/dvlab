import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.metrics import accuracy_score  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, StandardScaler  

# Load the dataset  
df = pd.read_csv("iris_dataset.csv")  
df.head()  
print("First few rows of the dataset:")  
print(df.head())  

# Define features and target  
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]  
Y = df["target"]  

# Split the data into training and testing sets  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)  

# Scale the features  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

# Encode the target labels  
encoder = LabelEncoder()  
Y_train_enc = encoder.fit_transform(Y_train)  
Y_test_enc = encoder.transform(Y_test)  

# Initialize the kNN classifier  
knn = KNeighborsClassifier(n_neighbors=3)  

# Fit the model  
knn.fit(X_train_scaled, Y_train_enc)  

# Make predictions  
Y_pred = knn.predict(X_test_scaled)  

# Calculate accuracy  
accuracy = accuracy_score(Y_test_enc, Y_pred)  
print(f"The KNN Classifier is {accuracy * 100:.0f}% accurate")  

# Visualization of the results  
labels = encoder.classes_  
markers = ["+", "o", "*"]  
colors = ["red", "blue", "gold"]  

for i, label in enumerate(labels):  
    class_points = (Y_pred == i)  
    plt.scatter(X_test_scaled[class_points, 0], X_test_scaled[class_points, 1],   
                label=f'Class {label}', marker=markers[i], color=colors[i])  

plt.title("KNN Classification Scatter Plot")  
plt.xlabel("Sepal Length (scaled)")  
plt.ylabel("Sepal Width (scaled)")  
plt.legend()  
plt.grid()  
plt.show()
