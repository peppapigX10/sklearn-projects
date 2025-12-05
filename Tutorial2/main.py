from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
housing = datasets.fetch_california_housing()

X = housing.data
y = housing.target

#initize polynomail features
poly = PolynomialFeatures()
X=poly.fit_transform(X)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

LR = LinearRegression()
GBR = HistGradientBoostingRegressor()
RFR = RandomForestRegressor(n_jobs=-1)

for i in [LR, GBR, RFR]:
    i.fit(X_train, y_train)

    y_pred = i.predict(X_test)

    print(i)
    #r2_score is for regressions, accuracy expects exact matches like pass/fail not $100 234 213.94, its for classification!
    accuracy = r2_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2%}")

print("How to optimize code shown in comments")
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# for j in [0.1, 0.05, 0.001]:
#     for i in [200, 250, 300]:
#         model = HistGradientBoostingRegressor(max_iter=i, learning_rate=j)
#         model.fit(X_train, y_train)
#         #joblib.dump(model, "my_model.joblib")
#         #at the end, local_model = joblib.load("filename")
#         y_pred = model.predict(X_test)

#         print(i, j)
# #r2_score is for regressions, accuracy expects exact matches like pass/fail not $100 234 213.94, its for classification!
#         accuracy = r2_score(y_test, y_pred)
#         print(f"Model Accuracy: {accuracy:.2%}")