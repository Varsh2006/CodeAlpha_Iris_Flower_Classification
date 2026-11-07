import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# LOAD AND EXPLORE THE DATASET
# ==========================================

# Load the Iris dataset
df = pd.read_csv("Iris.csv")

# Display basic information
print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpecies Distribution:")
print(df["Species"].value_counts())


# ==========================================
# DATA VISUALIZATION
# ==========================================

# Visualization 1: Species Count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Species")
plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")
plt.tight_layout()
plt.savefig("graphs/species_distribution.png")
plt.show()
plt.close()


# Visualization 2: Sepal Length vs Sepal Width
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="SepalWidthCm",
    hue="Species"
)
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.tight_layout()
plt.savefig("graphs/sepal_comparison.png")
plt.show()
plt.close()


# Visualization 3: Petal Length vs Petal Width
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.tight_layout()
plt.savefig("graphs/petal_comparison.png")
plt.show()
plt.close()


# ==========================================
# MACHINE LEARNING - IRIS CLASSIFICATION
# ==========================================

# Select input features
X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

# Select target variable
y = df["Species"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create the KNN model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix - Iris Classification")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png")
plt.show()
plt.close()


# ==========================================
# TEST WITH A NEW FLOWER
# ==========================================

# Example flower measurements
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict the species
prediction = model.predict(new_flower)

print("\nNew Flower Prediction:")
print("Predicted Species:", prediction[0])