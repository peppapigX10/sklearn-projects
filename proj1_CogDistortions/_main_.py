import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("data_CogD.csv")
X = df['Thought']
y = df['Cognitive_Distortion']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

text_vectorizer = TfidfVectorizer(max_features = 500, stop_words='english')
X_train = text_vectorizer.fit_transform(X_train)
X_test = text_vectorizer.transform(X_test)

model = LinearSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model type: {model}")
print(f"Model Accuracy: {accuracy:.2%}")


