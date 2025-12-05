import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Introductions
print("Logistic Regression - Will the student pass?")
print("_" * 32)


#Load your dataset
data = pd.read_csv(r"d:\My_Molly_Code\sklearn\LogisticRegression\data_students.csv")

# Control+/ to comment selected text # DEBUG BIG TIP :D
# print("\nColumns pandas sees:", data.columns.tolist())
# print("\nFirst 5 rows:")
# print(data.head())

#     #--> Remove hidden BOM characters
# data.columns = data.columns.str.strip()
# data.columns = data.columns.str.replace('\ufeff','')

#     #-->Pandas reads file as single column?
# pd.read_csv("data_student.csv", sep="\t") #for TSV
# pd.read_csv("data_student.csv") #default comma


#Continuing to load dataset
X = data[["Student_ID","Study_Hours_Per_Week","Attendance_Percentage","Previous_Grade","Assignments_Completed_Percentage"]]
y = data["Label"]

#Debugging
print(data.columns)
print("Columns pandas sees:", data.columns.tolist())

#Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#Predict
y_pred = model.predict(X_test)

#Evaluate
#Use this!!! v
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))
print("Classification Report:\n",classification_report(y_test, y_pred))


