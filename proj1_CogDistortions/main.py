import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("data_CogD.csv")

# print(df.columns.tolist())

X = df['Thought']
y = df['Cognitive_Distortion']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

textVectorizer = TfidfVectorizer(max_features = 500, stop_words='english')
X_train = textVectorizer.fit_transform(X_train)
X_test = textVectorizer.transform(X_test)

# model = LinearSVC(C=1)
# model.fit(X_train, y_train)
# #train, predict, check accuracy
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.2%}")

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")


while True:
    try:
        print("\n"+"-"*10+"Distorted Thoughts Classification AI"+"-"*10)
        X_user = input("💭 Thought, 'quit' to exit: ")
        if X_user.lower() == 'quit':
            print("Thank you for using the classifier!")
            break
        if not X_user.strip():
            continue
        X_user = textVectorizer.transform([X_user])

        #predict_probab returns PROBABILITIES not just prediction
        #[0] get the first or only row
        probabilities = model.predict_proba(X_user)[0]
        #using numpy sort the probabilities from highest to lowest
        sorted_indices = np.argsort(probabilities)[::-1]

        #Get first 3 indices or in other words top 3, indices are index
        top_3_indices = sorted_indices[:3]
        #for rank (position in loop) and index value
        #enumerate: adds a counter to each item
        #top_3_indices (list to loop through), starting from 1
        for rank, idx in enumerate(top_3_indices, 1):

            #get name of cogD at position idx
            distortion = label_encoder.classes_[idx]
            #Get value of position idx[index] = # in array of probailities
            probability = probabilities[idx]*100

            #100/5: 20 bar length (full), 82.5/5: 16 bar chars or bar length (portion of full or 20)
            bar_length = int(probability/5)
            bar = "█" * bar_length

            emoji = "🥇" if rank == 1 else "🥈" if rank==2 else "🥉"
            #Width: 22 characters, < means left-alignment
            #5.1f; 5: total width including decimal, .1 one decimal place, f: floating point number
            print(f"{emoji}{rank}. {distortion:<22}{probability:5.1f}% {bar}")
        # print(f"Cognitive Distortion: \n{y_user[0]}")
    except ValueError:
        print("Value error!")



