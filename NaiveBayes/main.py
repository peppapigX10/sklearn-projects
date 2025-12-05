import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC

#Load data
df = pd.read_csv("data_reviews.csv", sep=',')

print(df.columns.tolist())
#Separate features and target
X = df['Review']
y = df['Sentiment']

#Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

vectorizer_tfidf = TfidfVectorizer(max_features = 500, stop_words='english')
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

#------------Finding best model and hyper parameters-------------

# MNB = MultinomialNB()
# LG = LogisticRegression(max_iter=1000)
# SVM = LinearSVC()

# for i in [MNB, LG, SVM]:
#     i.fit(X_train_tfidf, y_train)

#     y_pred = i.predict(X_test_tfidf)
#     accuracy = accuracy_score(y_test, y_pred)
#     print("\n"+str(i))
#     print(f"Model Accuracy: {accuracy:.2%}")

# for j in [1.5, 2.0, 3.0,4.0]:
#     model = MultinomialNB(alpha=j)
#     model.fit(X_train_tfidf, y_train)
    
#     y_pred = model.predict(X_test_tfidf)
#     accuracy = accuracy_score(y_test, y_pred)
#     print("\n"+str(j))
#     print(f"Model accuracy: {accuracy:.2%}")

model = MultinomialNB(alpha=2.0)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy:.2%}")

