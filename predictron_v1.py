import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv("music.csv")

X = music_data.drop(columns=["genre"])
# X - is for input data (convention)
y = music_data["genre"]
# y - is for result data (convention)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier() # creation of a model object
model.fit(X_train, y_train) # trainig a model

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

aaa = 5
