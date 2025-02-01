import matplotlib.pyplot as plt  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  
import pandas as pd  

# Step 1: Load the dataset  
df = pd.read_csv("income_clustering.csv")  
df = df[["Age", "Income($)"]]  

# Step 2: Standardize the data  
scaler = StandardScaler()  
sc_df = scaler.fit_transform(df)  

# Step 3: Scatter plot before clustering  
plt.figure(figsize=(10, 5))  
plt.scatter(df["Age"], df["Income($)"], color="r", marker="*")  
plt.title("Age vs Income before Clustering")  
plt.xlabel("Age")  
plt.ylabel("Income($)")  
plt.show()  

# Step 4: Elbow Method for optimal number of clusters  
k_range = range(1, 11)  
sse = []  
for k in k_range:  
    kmn = KMeans(n_clusters=k, random_state=42)  
    kmn.fit(sc_df)  
    sse.append(kmn.inertia_)  

plt.figure(figsize=(10, 5))  
plt.plot(k_range, sse, color="r", marker=".")  
plt.title("Elbow Method")  
plt.xlabel("Number of Clusters")  
plt.ylabel("SSE")  
plt.xticks(k_range)  
plt.grid()  
plt.show()  

# Step 5: Fit K-means with the selected number of clusters (e.g., 3)  
optimal_clusters = 3  
kmn = KMeans(n_clusters=optimal_clusters, random_state=42)  
clusters = kmn.fit_predict(sc_df)  

df['clusters'] = clusters  # Add the cluster labels to the dataframe  

# Step 6: Visualize the clusters  
plt.figure(figsize=(10, 5))  
centroids = scaler.inverse_transform(kmn.cluster_centers_)  
cl1 = df[df['clusters'] == 0]  
cl2 = df[df['clusters'] == 1]  
cl3 = df[df['clusters'] == 2]  

plt.scatter(cl1['Age'], cl1['Income($)'], color="r", marker="s", label="Cluster 1")  
plt.scatter(cl2['Age'], cl2['Income($)'], color="b", marker="+", label="Cluster 2")  
plt.scatter(cl3['Age'], cl3['Income($)'], color="g", marker="v", label="Cluster 3")  
plt.scatter(centroids[:, 0], centroids[:, 1], label="Centroids", s=200, marker="*", color="black")  
plt.title("K-means Clustering of Income and Age Data")  
plt.xlabel("Age")  
plt.ylabel("Income($)")  
plt.legend()  
plt.grid()  
plt.show()
