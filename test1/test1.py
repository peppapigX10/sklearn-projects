# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the data
iris = load_iris()
X = iris.data
y = iris.target

# Convert it to a readable table
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Show the first 10 rows
print(df.head(10))

# Save it as a CSV file
df.to_csv('iris_data.csv', index=False)
print("Data saved to iris_data.csv!")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")