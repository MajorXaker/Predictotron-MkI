import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


music_data = pd.read_csv("music.csv")

# X = music_data.drop(columns=["genre"])
# # X - is for input data (convention)
# y = music_data["genre"]
# # y - is for result data (convention)
#
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# model = DecisionTreeClassifier()
# model.fit(X, y)



# predictions = model.predict([[21, 1]])

# saved = joblib.dump(model, 'music-recommender.joblib')

model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])

aaa = 5
