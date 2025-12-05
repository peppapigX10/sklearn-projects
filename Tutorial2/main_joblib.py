from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import joblib
housing = datasets.fetch_california_housing()

X = housing.data
y = housing.target

#initize polynomail features
poly = PolynomialFeatures()
poly.fit_transform(X)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(max_iter=200)
model.fit(X_train, y_train)

joblib.dump(model, "my_model.joblib")

local_model = joblib.load("my_model.joblib")

y_pred = local_model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")