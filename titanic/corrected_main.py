
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
train_data = pd.read_csv("C:/Users/boetr/Desktop/Programeren/titanic/train.csv")
test_data = pd.read_csv("C:/Users/boetr/Desktop/Programeren/titanic/test.csv")

test_data['Survived'] = 0


# EDA: Visualize the first few rows of the dataset
print("First few rows of the dataset:")
print(train_data.head())

# EDA: Basic statistics of the dataset
print("Basic statistics of the dataset:")
print(train_data.describe())

# EDA: Count of null values in each column
print("Count of null values in each column:")
print(train_data.isnull().sum())

# Data Preprocessing: Handle missing values
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# EDA: Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train_data.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Feature Engineering: Drop irrelevant columns
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Data Preprocessing: Convert categorical columns to numerical
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)

# Split the data into training and test sets
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

