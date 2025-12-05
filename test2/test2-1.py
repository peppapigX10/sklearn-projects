# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
import pandas as pd # for working with tables

# Load and train the model
iris = load_iris() # Load iris data into var iris, var iris is a container with multiple compartments (iris.data & iris.target)
X = iris.data # iris.data contains inputs or x-values  in other words is the description 
y = iris.target # iris.target contains outputs or y-values, y is lowercase and X is uppercase is a tradition followed by everyone in math and ML

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #split data to 4 parts, use t_t_s, 
#put in the X and y values, use 0.3 or 30% for testing, 70% for training, random_state controls how data is shuffled, # =shuffled same way, any number works each with diff accuracy, no number is better
#shuffling the data the same way causes easiness for debugging, so you get the same accuracy everytime
#X_train (105 flower mesurements)
#y_train (105 species labels)
#X_test (45 flower measurements)
#y_test (45 species labels)

model = DTC #use DTC model, the model is the brain
model.fit(X_train, y_train) #train the model


