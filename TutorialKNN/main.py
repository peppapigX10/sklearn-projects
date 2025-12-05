import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('500hits.csv',encoding='latin-1')
df = df.drop(columns=['PLAYER','CS'])

print(df.head())

X = df.iloc[:,0:13]
y = df.iloc[:,13]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform (X_train)
X_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(y_pred)

accuracy=knn.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")

print("\nConfusion Matrix\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report\n",classification_report(y_test, y_pred))

