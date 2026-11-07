# Iris Flower Classification

## Project Overview

The objective of this project is to build a machine learning classification model that predicts the species of an Iris flower based on its sepal and petal measurements.

The project uses the Iris dataset containing measurements of three Iris species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Objective

The main objectives of this project are:

- Analyze the Iris flower dataset.
- Check the dataset for missing values.
- Visualize the distribution and relationships between flower measurements.
- Train a machine learning classification model.
- Evaluate the model using accuracy and classification metrics.
- Predict the species of a new Iris flower.

## Dataset

The dataset contains **150 flower samples** and the following columns:

| Column | Description |
|---|---|
| Id | Unique identifier |
| SepalLengthCm | Sepal length in centimeters |
| SepalWidthCm | Sepal width in centimeters |
| PetalLengthCm | Petal length in centimeters |
| PetalWidthCm | Petal width in centimeters |
| Species | Iris flower species |

The dataset contains 50 samples of each species.

No missing values were found in the dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Model

The project uses the **K-Nearest Neighbors (KNN)** classification algorithm.

The four flower measurements used as input features are:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The `Id` column is not used for model training because it is only an identifier.

The dataset was divided into:

- **80% training data**
- **20% testing data**

## Data Visualization

The project includes the following visualizations:

1. Distribution of Iris Species
2. Sepal Length vs Sepal Width
3. Petal Length vs Petal Width
4. Confusion Matrix

The generated graphs are available in the `graphs` folder.

## Model Performance

The KNN model achieved:

**Test Accuracy: 100%**

The classification report showed 1.00 precision, recall, and F1-score for all three Iris species on the test set.

The model correctly classified all 30 test samples in the selected test split.

## New Flower Prediction

The trained model was also tested with a new set of flower measurements:

- Sepal Length: 5.1 cm
- Sepal Width: 3.5 cm
- Petal Length: 1.4 cm
- Petal Width: 0.2 cm

The model predicted the species as:

**Iris-setosa**

## Project Structure

```text
CodeAlpha_Iris_Flower_Classification/
│
├── Iris.csv
├── iris_classification.py
├── requirements.txt
├── README.md
│
└── graphs/
    ├── species_distribution.png
    ├── sepal_comparison.png
    ├── petal_comparison.png
    └── confusion_matrix.png
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd CodeAlpha_Iris_Flower_Classification
```

### 3. Install the required libraries

```bash
py -m pip install -r requirements.txt
```

### 4. Run the project

```bash
py iris_classification.py
```

