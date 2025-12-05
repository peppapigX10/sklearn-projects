import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("-"*17+"DIAGNOSIS AI MODEL"+"-"*17)
#1.Load dataset
data=pd.read_csv("data_patients.csv")
print(data.head())

#2.Separate features and target
print("\nColumns pandas sees:", data.columns.tolist())

X=data[['Age', 'BloodPressure', 'Cholesterol', 'HeartRate', 'Glucose']]
y=data['Label']

#Optional for KNN standardize features for KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#3. Split data into training and testing sets
X_train, X_test, y_train, y_test=train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

#4. Create and train KNN model

#n_neighbors decides how many nearby samples or neighbors the algorithm will look at when deciding the class of a new data point
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#5. Make predictions
y_pred=knn.predict(X_test)

#6. Evaluate the model
accuracy=accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))